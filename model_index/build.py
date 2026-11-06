"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink-model-index and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

One indexing run: find every ref, give each its ONNX, publish the index.

A ref resolves by the most certain evidence it has (METHODS, best first):
- override: overrides.json says so (a human looked);
- tree: the commit has big_driving_supercombo.onnx, and its pointer is the answer;
- weights: the commit has only the precompiled pkl, and the pkl carries the
  weights of one ONNX in the library (weights.py);
- weights-sunnypilot: comma no longer serves that pkl, so sunnypilot's
  recompiled build of it is matched instead;
- checkpoint: the model_checkpoint the pkl's metadata names, until a run has
  the budget to check its weights;
- message: no bytes to check; the ONNX comma's commit or pull request deletes,
  or the export their subjects name.
The first four are final; a ref resolved by the last two is tried again.

The library is every ONNX comma has published: each export in its HuggingFace
model repo, now or in its history, and each ONNX any known commit has in its
tree. Only the current exports name a checkpoint or a message's export: a
replaced one shares its path with what replaced it. An ONNX is
downloaded once and only its fingerprint kept; a pkl is downloaded once and
matched, and again only when the library grows. A commit's tree and messages
never change, so each is read once and kept in state.json: a quiet hour costs
a handful of requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from model_index import lfs, net, sources, weights

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'state.json'
LIBRARY = ROOT / 'library' / 'onnx'
INDEX = ROOT / 'index' / 'v1.json'
OVERRIDES = ROOT / 'overrides.json'
CACHE = ROOT / '.cache' / 'sources.json'   # not committed: kept between runs by actions/cache
REPORT = ROOT / 'report.md'                # what needs a human; the workflow's issue
NOTES = ROOT / 'notes.md'                  # everything else worth a look
SUMMARY = ROOT / 'summary.txt'
SCHEMA = 1
METHODS = ('override', 'tree', 'weights', 'weights-sunnypilot', 'checkpoint', 'message')
FINAL = frozenset(METHODS[:4])
RESULT = ('status', 'reason', 'oid', 'size', 'method', 'ambiguous', 'checkpoint_disagrees')
ANCESTORS = 40
GONE_RECHECK = 7 * 86400
# a ref still unchecked by its weights this long after it turned up needs a human: something upstream moved
STALE = 2 * 86400
PROBE_THREADS = 16
PR_FOLDER = 'comma Pull Requests'
# what a listed model carries of its catalog entry, all a catalog in sunnypilot's shape needs
CATALOG_FIELDS = ('short_name', 'is_20hz', 'index', 'minimum_selector_version', 'build_time', 'generation', 'environment', 'runner')
# what a pull request's model takes from the newest catalog model: the runtime it was built for
RUNTIME_FIELDS = ('is_20hz', 'generation', 'environment', 'runner')
COMMIT_PATCH_URL = 'https://github.com/commaai/openpilot/commit/{ref}.patch'
PULL_PATCH_URL = 'https://github.com/commaai/openpilot/pull/{number}.patch'
PATCH_MAX = 1 << 20
# how comma's subjects name an export: the whole uuid, or its first eight and the checkpoint ('f22b8b34/16384')
_EXPORT = re.compile(r'\b([0-9a-f]{8})(?:-[0-9a-f]{4}-|/)')
_PULL = re.compile(r'\(#(\d+)\)$')
_ONNX_DIFF = re.compile(r'^diff --git a/\S*selfdrive/modeld/models/' + re.escape(lfs.BIG_ONNX) + r' ')

log = logging.getLogger('model_index')


def _read(path: Path, default):
  try:
    return json.loads(path.read_text())
  except FileNotFoundError:
    return default


def _write(path: Path, data) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=1, sort_keys=True) + '\n')


def settle(entry: dict, status: str, **fields) -> None:
  """Replace a ref's result whole, so nothing from an earlier one lingers."""
  for k in RESULT:
    entry.pop(k, None)
  entry.update({'status': status, **{k: v for k, v in fields.items() if v is not None}})


def patch_subjects(text: str) -> list[str]:
  """Every commit's subject in a patch, in order: one for a commit's, one per
  commit for a pull request's."""
  lines = text.splitlines()
  subjects = []
  for i, line in enumerate(lines):
    if line.startswith('Subject:'):
      subject = [line.removeprefix('Subject:').strip()]
      # a long subject is folded onto indented lines; the headers end at a blank one
      for cont in lines[i + 1:]:
        if not cont[:1].isspace() or not cont.strip():
          break
        subject.append(cont.strip())
      subjects.append(re.sub(r'^\[PATCH[^\]]*\]\s*', '', ' '.join(subject)))
  return subjects


def diff_pointer(text: str) -> lfs.Pointer | None:
  """The big ONNX's pointer in a patch's diffs, the last one wins: the pointer a
  diff gives the file, or for a diff that deletes it, the pointer it had. A
  commit that swaps the ONNX for a pkl deletes the ONNX the pkl was built from."""
  found, section = None, None

  def close():
    nonlocal found
    if section is not None:
      added = lfs.parse_pointer('\n'.join(x[1:] for x in section if x.startswith('+') and not x.startswith('+++')))
      removed = lfs.parse_pointer('\n'.join(x[1:] for x in section if x.startswith('-') and not x.startswith('---')))
      found = added or removed or found

  for line in text.splitlines():
    # each diff, and each commit of a pull request's patch, ends the one before
    if line.startswith(('diff --git ', 'From ')):
      close()
      section = [] if _ONNX_DIFF.match(line) else None
    elif section is not None:
      section.append(line)
  close()
  return found


def _catalog_rank(name: str) -> tuple[bool, int]:
  """Newer catalogs first, chestnut ones before the rest."""
  m = sources.CATALOG.match(name + '.json')
  return (bool(m.group(1)), int(m.group(2))) if m else (False, -1)


def _folder(listing: dict) -> str | None:
  # catalogs have kept overrides as a dict and as a list of {key, value}
  o = listing.get('overrides')
  if isinstance(o, dict):
    return o.get('folder')
  return next((x.get('value') for x in o or [] if isinstance(x, dict) and x.get('key') == 'folder'), None)


class Run:
  def __init__(self, work: Path, max_bytes: int, recheck: bool):
    self.work, self.budget, self.recheck = work, max_bytes, recheck
    self.now = int(time.time())
    self.state = _read(STATE, {})
    self.refs: dict[str, dict] = self.state.setdefault('refs', {})
    self.pkls: dict[str, dict] = self.state.setdefault('pkls', {})
    self.hf_commits: dict[str, dict] = self.state.setdefault('hf_commits', {})  # commit -> its tree's ONNX
    self.cache = _read(CACHE, {})
    self.cache.setdefault('catalogs', {})
    self.cache.setdefault('prs', {})
    overrides = _read(OVERRIDES, {})
    self.overrides, self.folders = overrides.get('refs', {}), overrides.get('folders', {})
    # the model a Jetson runs with nothing picked, by its ref
    self.default: str | None = overrides.get('default')
    # refs a human settled outright; an `accept` override only quiets the report
    self.settled = {ref for ref, o in self.overrides.items() if 'unavailable' in o or 'oid' in o}
    self.library = {p.stem: _read(p, None) for p in sorted(LIBRARY.glob('*.json'))}
    self.exports: dict[str, dict] = {}
    self.notes: list[str] = []

  def take(self, n: int) -> bool:
    if n > self.budget:
      return False
    self.budget -= n
    return True

  def guard(self, what: str, fn, *args) -> bool:
    try:
      fn(*args)
      return True
    except (net.NetworkError, net.NotFound, ValueError, KeyError) as e:
      self.notes.append(f'{what}: {e}')
      return False

  # --- what was found ----------------------------------------------------

  def merge(self, found: dict[str, sources.Seen], listings_fresh: bool) -> None:
    """Fold this run's sources into state. Today's catalogs decide what is
    listed (unless they could not be read); everything ever seen stays."""
    if listings_fresh:
      for e in self.refs.values():
        e.pop('listing', None)
    heads = {seen.pr: ref for ref, seen in found.items() if seen.pr}
    for ref, e in self.refs.items():
      if e.get('pr') in heads and heads[e['pr']] != ref:
        e['open'] = False  # force-pushed: only a PR's current head is listed
    for ref, seen in found.items():
      e = self.refs.setdefault(ref, {})
      e['names'] = sorted({*e.get('names', []), *seen.names})
      e['sources'] = sorted({*e.get('sources', []), *seen.sources})
      if seen.listing:
        e['listing'] = seen.listing
      if seen.artifacts:
        e['artifacts'] = sorted({*map(tuple, e.get('artifacts', [])), *seen.artifacts})
      if seen.pr:
        e['pr'], e['open'] = seen.pr, seen.open
    for e in self.refs.values():
      e.setdefault('since', self.now)  # when it turned up, for STALE

  # --- the library -------------------------------------------------------

  def add_onnx(self, oid: str, size: int, source: str, export: str | None = None) -> None:
    entry = self.library.get(oid)
    if entry is None:
      if not self.take(size):
        self.notes.append(f'deferred ONNX {oid[:16]} ({source}): download budget spent')
        return
      path = lfs.download([lfs.Pointer(oid, size)], self.work / f'{oid}.onnx')
      try:
        entry = {'size': size, 'export': None, 'sources': [], 'tensors': weights.fingerprint(path)}
      finally:
        path.unlink(missing_ok=True)
      log.warning('fingerprinted %s (%s)', oid[:16], source)
    new = {**entry, 'export': export or entry['export'], 'sources': sorted({*entry['sources'], source})}
    if new != self.library.get(oid):
      self.library[oid] = new
      _write(LIBRARY / f'{oid}.json', new)

  def digest(self) -> str:
    return hashlib.sha256(''.join(sorted(self.library)).encode()).hexdigest()[:16]

  def export_oids(self, prefix: str) -> list[str]:
    return [o for o, x in self.exports.items() if x['export'].startswith(prefix)]

  # --- facts read once per ref -------------------------------------------

  @staticmethod
  def probe(ref: str) -> dict:
    """What the commit's tree holds: the ONNX's pointer, the pkl's, or neither."""
    if found := lfs.pointer_at(ref, lfs.BIG_ONNX):
      return {'onnx': [found[0].oid, found[0].size, found[1]]}
    pointers = lfs.pkl_at(ref)
    return {'pkl': [[p.oid, p.size, p.url] for p in pointers]} if pointers else {'none': True}

  def probe_all(self) -> None:
    def keep(ref, future):
      self.refs[ref]['tree'] = future.result()

    todo = [ref for ref, e in self.refs.items() if 'tree' not in e and ref not in self.settled]
    with ThreadPoolExecutor(PROBE_THREADS) as ex:
      for ref, future in [(ref, ex.submit(self.probe, ref)) for ref in todo]:
        self.guard(f'`{ref}`', keep, ref, future)

  def messages(self, ref: str) -> dict:
    """The export ids comma's subjects name (newest mention first) and the ONNX
    pointer its diffs delete, from the commit's patch and its pull request's.
    Patches are not API calls."""
    e = self.refs[ref]
    if 'messages' not in e:
      patch = net.get(COMMIT_PATCH_URL.format(ref=ref), limit=PATCH_MAX).decode('utf-8', 'replace')
      subjects, diff = patch_subjects(patch), diff_pointer(patch)
      numbers = {str(e['pr'])} if e.get('pr') else set()
      numbers |= {m.group(1) for m in map(_PULL.search, subjects[:1]) if m}
      for number in sorted(numbers):
        pull = net.get(PULL_PATCH_URL.format(number=number), limit=PATCH_MAX).decode('utf-8', 'replace')
        subjects += patch_subjects(pull)
        diff = diff or diff_pointer(pull)
      e['messages'] = {'ids': list(dict.fromkeys(reversed(_EXPORT.findall(' '.join(subjects))))),
                       'diff': [diff.oid, diff.size] if diff else None}
    return e['messages']

  def ancestors(self, ref: str) -> list[str]:
    """The ONNX of the nearest resolved first-parent ancestors, nearest first:
    a recompile keeps the weights of the model before it."""
    try:
      commits = net.github(f'repos/{sources.COMMA}/commits?sha={ref}&per_page={ANCESTORS + 1}')
    except (net.NetworkError, net.NotFound):
      return []
    oids = [self.refs.get(c['sha'], {}).get('oid') for c in commits[1:]]
    return list(dict.fromkeys(o for o in oids if o))

  # --- resolving ---------------------------------------------------------

  def resolve(self, ref: str) -> None:
    e = self.refs[ref]
    if ref in self.settled:
      o = self.overrides[ref]
      if 'unavailable' in o:
        settle(e, 'unavailable', reason=o['unavailable'])
      else:
        settle(e, 'resolved', oid=o['oid'], size=int(o['size']), method='override')
      return
    if (e.get('status') == 'resolved' and e.get('method') in FINAL and not self.recheck) or 'tree' not in e:
      return
    tree = e['tree']
    if 'onnx' in tree:
      settle(e, 'resolved', oid=tree['onnx'][0], size=tree['onnx'][1], method='tree')
    elif 'pkl' in tree:
      self.resolve_pkl(ref, [lfs.Pointer(*p) for p in tree['pkl']])
    else:
      settle(e, 'unresolved', reason='no big model ONNX or pkl at this commit')

  def resolve_pkl(self, ref: str, pointers: list[lfs.Pointer]) -> None:
    """Weigh the evidence in order: comma's pkl, each sunnypilot build of it
    (matched by weights), the pkl's checkpoint, then comma's messages."""
    e = self.refs[ref]
    key = lfs.pkl_key(pointers)
    comma = self.pkls.setdefault(key, {})
    if 'checkpoint' not in comma and time.time() - comma.get('gone_at', 0) > GONE_RECHECK:
      try:
        comma['checkpoint'] = lfs.checkpoint(lfs.head(pointers[0]))
        comma.pop('gone_at', None)
      except net.NetworkError:
        comma['gone_at'] = int(time.time())  # a force-pushed PR head, a reverted commit
    checkpoint = comma.get('checkpoint') or []
    named = next((oids[0] for pair in reversed(checkpoint) if (oids := self.export_oids(pair + '/'))), None)

    evidence = [] if 'gone_at' in comma else [('weights', pointers, key)]
    evidence += [('weights-sunnypilot', [lfs.Pointer(sha, 0, url)], f'sunnypilot:{sha}') for url, sha in e.get('artifacts', [])]
    short = False
    for method, parts, k in evidence:
      cached = self.pkls.setdefault(k, {})
      if not ('full' in cached and not self.recheck and (cached['full'] or cached.get('library') == self.digest())):
        if not self.take(sum(p.size for p in parts) or 1 << 30):
          short = True
          break
        try:
          cached.update(full=self.match(parts, k), library=self.digest())
        except net.NetworkError as err:
          self.notes.append(f'`{ref}` {method}: {err}')
          if method == 'weights':
            comma['gone_at'] = int(time.time())
          continue
      priors = [named] if named else []
      oid, ties = weights.pick(cached['full'], priors)
      if ties:
        oid, ties = weights.pick(cached['full'], priors + self.ancestors(ref))
      if oid:
        settle(e, 'resolved', oid=oid, size=self.library[oid]['size'], method=method, ambiguous=ties or None,
               checkpoint_disagrees=named if named and named != oid else None)
        return

    if named and (short or named not in self.library):
      settle(e, 'resolved', oid=named, size=self.exports[named]['size'], method='checkpoint')
    elif named:
      settle(e, 'unresolved', reason=f'its pkl names checkpoint {checkpoint[-1]}, but does not carry that ONNX\'s weights')
    elif short:
      self.resolve_by_message(ref, 'pending', 'download budget spent')
    else:
      why = 'its weights match no known ONNX' if evidence else 'comma no longer serves its pkl and sunnypilot never built it'
      if checkpoint:
        why += f'; it was built from checkpoint {checkpoint[-1]}, which comma has not published'
      self.resolve_by_message(ref, 'unresolved', why)

  def match(self, parts: list[lfs.Pointer], key: str) -> dict:
    path = lfs.download(parts, self.work / f'{key.split(":")[-1]}.pkl')
    try:
      return weights.match(path, {o: x['tensors'] for o, x in self.library.items()})
    finally:
      path.unlink(missing_ok=True)

  def resolve_by_message(self, ref: str, status: str, why: str) -> None:
    """The last resort, with no bytes to check: the ONNX comma's diffs delete,
    or the published export its subjects name, newest mention first."""
    e = self.refs[ref]
    try:
      msg = self.messages(ref)
    except (net.NetworkError, net.NotFound) as err:
      settle(e, status, reason=f'{why}, and its commit could not be read: {err}')
      return
    named = [o for export in msg['ids'] for o in self.export_oids(export)]
    missing = [export for export in msg['ids'] if not self.export_oids(export)]
    if msg['diff']:
      settle(e, 'resolved', oid=msg['diff'][0], size=msg['diff'][1], method='message')
      # into the library, so the next run can check this and every other pkl against it by weights
      self.add_onnx(msg['diff'][0], msg['diff'][1], f'diff:{ref}')
    elif named:
      settle(e, 'resolved', oid=named[0], size=self.exports[named[0]]['size'], method='message',
             ambiguous=list(dict.fromkeys(named)) if len(set(named)) > 1 else None)
    elif missing:
      settle(e, status, reason=f'{why}; it names export {", ".join(missing)}, which comma has not published')
    else:
      settle(e, status, reason=why)

  # --- publishing --------------------------------------------------------

  def models(self, resolved: set[str]) -> list[dict]:
    """What a Jetson can pick: one entry per ONNX, under the newest catalog
    entry that has it, in that entry's folder, with the entry's CATALOG_FIELDS.
    An open comma pull request whose ONNX nothing else lists goes in
    PR_FOLDER, with the newest catalog model's RUNTIME_FIELDS. Every entry has
    a short name."""
    best: dict[str, tuple] = {}
    for ref, e in self.refs.items():
      if ref not in resolved:
        continue
      if e.get('listing'):
        catalog, listing = max(e['listing'].items(), key=lambda kv: (_catalog_rank(kv[0]), int(kv[1].get('index', 0))))
        model = {'name': listing.get('display_name') or ref[:10], 'folder': _folder(listing), 'catalog': catalog,
                 **{k: listing[k] for k in CATALOG_FIELDS if k in listing}}
        rank = (1, _catalog_rank(catalog), int(listing.get('index', 0)))
      elif e.get('pr') and e.get('open'):
        name = next((n for n in e.get('names', []) if n.startswith(f'#{e["pr"]} ')), f'#{e["pr"]}')
        model, rank = {'name': name, 'folder': PR_FOLDER, 'pr': e['pr'], 'short_name': f'#{e["pr"]}'}, (0, (False, 0), e['pr'])
      else:
        continue
      model = {'ref': ref, 'oid': e['oid'], 'size': e['size'], **model}
      model['folder'] = self.folders.get(ref) or model['folder'] or 'Other Models'
      model['short_name'] = model.get('short_name') or ref[:7]
      if e['oid'] not in best or rank > best[e['oid']][0]:
        best[e['oid']] = (rank, model)
    ranked = [m for _, m in sorted(best.values(), key=lambda rm: rm[0], reverse=True)]
    newest = next((m for m in ranked if 'pr' not in m), {})
    for m in ranked:
      if 'pr' in m:
        m.update({k: newest[k] for k in RUNTIME_FIELDS if k in newest})
    folders = list(dict.fromkeys(m['folder'] for m in ranked))
    return sorted((m for _, m in best.values()), key=lambda m: (m['folder'] == PR_FOLDER, folders.index(m['folder']),
                                                                 -int(m.get('index', m.get('pr', 0)))))

  def index(self) -> dict:
    refs, unresolved = {}, {}
    for ref, e in sorted(self.refs.items()):
      if e.get('status') == 'resolved':
        refs[ref] = {'oid': e['oid'], 'size': e['size'], 'method': e['method'], **({'ambiguous': True} if e.get('ambiguous') else {})}
      else:
        unresolved[ref] = e.get('reason') or e.get('status') or 'not looked at yet'
    onnx = {o: {'size': x['size'], 'export': x.get('export')} for o, x in sorted(self.library.items())}
    models = self.models(set(refs))
    return {'schema': SCHEMA, 'default': self.listed_default(models, refs), 'refs': refs, 'onnx': onnx, 'models': models,
            'unresolved': unresolved}

  def listed_default(self, models: list[dict], refs: dict) -> str | None:
    """overrides.json's default as the list names it: its own entry, or the
    one listing its ONNX. None when it lists neither, which reports() says."""
    oid = (refs.get(self.default) or {}).get('oid')
    return next((m['ref'] for m in models if m['ref'] == self.default), None) or next((m['ref'] for m in models if m['oid'] == oid), None)

  def reports(self) -> tuple[str, str]:
    """(what needs a human, everything else worth a look)."""
    human, info = [], []
    if self.default and self.listed_default(self.models(set(self.refs)), {r: e for r, e in self.refs.items() if e.get('oid')}) is None:
      human.append(f'- the default `{self.default}` (overrides.json) is not listed: a Jetson with nothing picked runs its own fallback')
    for ref, e in sorted(self.refs.items()):
      name = ', '.join(e.get('names', [])[:2]) or '(no name)'
      stale = ref not in self.overrides and self.now - e.get('since', self.now) > STALE
      days = f' for {(self.now - e.get("since", self.now)) // 86400} days' if stale else ''
      if e.get('status') == 'unresolved':
        human.append(f'- `{ref}` {name}: {e.get("reason", "")}')
      elif e.get('checkpoint_disagrees'):
        human.append(f'- `{ref}` {name}: its weights are {e["oid"][:12]}, but its pkl names checkpoint ONNX {e["checkpoint_disagrees"][:12]}')
      elif e.get('status') == 'pending':
        (human if stale else info).append(f'- `{ref}` {name}: pending{days}, {e.get("reason", "")}')
      elif e.get('method') in ('checkpoint', 'message'):
        (human if stale else info).append(f'- `{ref}` {name}: {e["oid"][:12]} by {e["method"]} only, not yet checked by weights{days}'
                                          + (f' (also names {", ".join(o[:12] for o in e["ambiguous"])})' if e.get('ambiguous') else ''))
      elif e.get('ambiguous'):
        info.append(f'- `{ref}` {name}: weights fit {len(e["ambiguous"])} ONNX equally ({", ".join(o[:12] for o in e["ambiguous"])}), took {e["oid"][:12]}')
    return '\n'.join(human), '\n'.join(info + [f'- {n}' for n in self.notes])


def main(argv=None) -> int:
  ap = argparse.ArgumentParser(description='Resolve every big comma driving model to its ONNX and publish index/v1.json.')
  ap.add_argument('--history', action='store_true', help='also read every past catalog and all of comma master')
  ap.add_argument('--recheck', action='store_true', help='match every pkl again, even ones already matched')
  ap.add_argument('--max-gb', type=float, default=60.0, help='download budget for this run')
  ap.add_argument('--work', type=Path, default=None, help='scratch directory for downloads')
  args = ap.parse_args(argv)
  logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

  with tempfile.TemporaryDirectory(dir=args.work) as tmp:
    run = Run(Path(tmp), int(args.max_gb * (1 << 30)), args.recheck)
    found: dict[str, sources.Seen] = {}
    listings_fresh = run.guard('sunnypilot catalogs', sources.sunnypilot, found, run.cache['catalogs'])
    run.guard('jetlink catalog', sources.jetlink, found)
    run.guard('comma master', sources.comma_master, found, args.history)
    run.guard('comma pull requests', sources.comma_prs, found, run.cache['prs'], 3 if args.history else 1)
    if args.history:
      run.guard('catalog history', sources.history, found, Path(tmp))
    run.merge(found, listings_fresh)

    run.exports = sources.huggingface_exports()
    history: dict[str, dict] = {}
    run.guard('HuggingFace history', sources.huggingface_history, history, run.exports, run.hf_commits)
    for tag, exports in (('huggingface', run.exports), ('huggingface-history', history)):
      for oid, x in exports.items():
        run.guard(f'export {oid[:16]}', run.add_onnx, oid, x['size'], f'{tag}:{x["export"]}', x['export'])
    run.probe_all()
    # every ONNX in a tree joins the library before any pkl is matched against it
    for ref, e in sorted(run.refs.items()):
      if 'onnx' in e.get('tree', {}):
        run.guard(f'`{ref}`', run.add_onnx, e['tree']['onnx'][0], e['tree']['onnx'][1], f'tree:{ref}')
    for ref in sorted(run.refs):
      run.guard(f'`{ref}`', run.resolve, ref)

  index = run.index()
  _write(STATE, run.state)
  _write(CACHE, run.cache)
  _write(INDEX, index)
  human, info = run.reports()
  REPORT.write_text(human + '\n' if human else '')
  NOTES.write_text(info + '\n' if info else '')
  summary = f'{len(index["refs"])} refs, {len(index["onnx"])} ONNX, {len(index["models"])} listed, {len(index["unresolved"])} unresolved'
  SUMMARY.write_text(summary + '\n')
  print(summary)
  print(human)
  print(info)
  return 0


if __name__ == '__main__':
  sys.exit(main())
