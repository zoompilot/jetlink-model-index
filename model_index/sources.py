"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink-model-index and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Where big driving models turn up:

- comma's pull requests that touch the big model, before they merge (Light
  Speed was a PR head when sunnypilot listed it, and was force-pushed after);
- comma's master, every commit that changed the big ONNX or pkl;
- comma's model repo on HuggingFace, the exports behind precompiled pkls;
- sunnypilot's model catalogs (every version, and with `history`, every
  version of every one it ever published);
- jetlink's own extra catalog, whose models stay resolvable but are no
  longer listed.

Each yields refs, comma commits, with whatever name and listing fields it gives.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from model_index import lfs, net

COMMA = 'commaai/openpilot'
SP_MODELS = 'sunnypilot/sunnypilot-models'
SP_RAW = f'https://raw.githubusercontent.com/{SP_MODELS}/refs/heads/gh-pages/docs/{{name}}'
CATALOG = re.compile(r'driving_models_(chestnut_)?v(\d+)\.json$')
JETLINK = 'zoompilot/jetlink'
JETLINK_EXTRA = f'https://raw.githubusercontent.com/{JETLINK}/refs/heads/main/catalog/extra_big_models.json'
HF_TREE = f'https://huggingface.co/api/models/{lfs.DRIVING_MODELS_REPO}/tree/main?recursive=true'
_REF = re.compile(r'[0-9a-f]{40}')
# what a catalog entry carries that a model list needs
LISTING = ('short_name', 'display_name', 'is_20hz', 'index', 'minimum_selector_version', 'overrides', 'build_time', 'environment',
           'generation', 'runner')


@dataclass
class Seen:
  names: set[str] = field(default_factory=set)
  sources: set[str] = field(default_factory=set)
  listing: dict = field(default_factory=dict)   # catalog name -> its entry's listing fields
  artifacts: set[tuple[str, str]] = field(default_factory=set)  # (url, sha256) of sunnypilot's builds of it
  pr: int | None = None   # the comma pull request this is the head of
  open: bool = False


def _add(found: dict[str, Seen], ref, source: str, name: str | None = None) -> Seen | None:
  if not isinstance(ref, str) or not _REF.fullmatch(ref):
    return None
  seen = found.setdefault(ref, Seen())
  seen.sources.add(source)
  if name:
    seen.names.add(name)
  return seen


def _artifacts(bundle: dict) -> list[list[str]]:
  """[url, sha256] of each driving pkl sunnypilot built for a bundle; the
  catalog has spelled these fields two ways."""
  found = []
  for m in bundle.get('models') or []:
    a = (m or {}).get('artifact') or {}
    uri = a.get('download_uri') or a.get('downloadUri') or {}
    url, sha = uri.get('url') or uri.get('uri'), uri.get('sha256')
    if isinstance(url, str) and url.startswith('https://') and lfs.SHA256.fullmatch(str(sha)) and 'driving' in url and 'dmonitoring' not in url:
      found.append([url, sha])
  return found


def _bundles(data, big_only: bool) -> list[dict]:
  """A catalog's entries, cut to what is kept: ref, listing fields, artifacts.
  Its `pointers` (jetlink's) become entries with only a ref."""
  if not isinstance(data, dict):
    return []
  kept = [{'ref': b.get('ref'), 'listing': {k: b[k] for k in LISTING if k in b}, 'artifacts': _artifacts(b)}
          for b in data.get('bundles') or [] if isinstance(b, dict) and (b.get('is_big') or not big_only)]
  return kept + [{'ref': r} for r in data.get('pointers') or {}]


def _catalog(found: dict[str, Seen], bundles: list[dict], source: str, catalog: str | None = None) -> None:
  for b in bundles:
    listing = b.get('listing') or {}
    if seen := _add(found, b['ref'], source, listing.get('display_name')):
      if catalog and listing:
        seen.listing[catalog] = listing
      seen.artifacts.update(map(tuple, b.get('artifacts') or []))


def _sunnypilot_big_only(name: str) -> bool:
  # chestnut catalogs are all big models; the others only where a bundle says so
  return 'chestnut' not in name


def sunnypilot(found: dict[str, Seen], cache: dict) -> None:
  """Every catalog sunnypilot serves now. `cache` keeps each one's cut-down
  entries by blob sha, so only a catalog that changed is fetched again."""
  listed = {e['name']: e['sha'] for e in net.github(f'repos/{SP_MODELS}/contents/docs?ref=gh-pages') if CATALOG.match(e.get('name', ''))}
  for name in set(cache) - set(listed):
    del cache[name]
  for name, sha in sorted(listed.items()):
    # cut to LISTING: a cache cut to other fields is fetched again
    if cache.get(name, {}).get('sha') != sha or cache[name].get('fields') != list(LISTING):
      cache[name] = {'sha': sha, 'fields': list(LISTING), 'bundles': _bundles(net.get_json(SP_RAW.format(name=name)), _sunnypilot_big_only(name))}
    _catalog(found, cache[name]['bundles'], f'sunnypilot:{name}', name.removesuffix('.json'))


def _git_blobs(repo: Path, pathspec: str):
  """The contents of every version any commit gave a file under `pathspec`."""
  log = subprocess.run(['git', '-C', str(repo), 'log', '--all', '--format=', '--raw', '--no-abbrev', '--', pathspec],
                       capture_output=True, text=True, check=True).stdout
  blobs: dict[str, str] = {}
  for line in log.splitlines():
    if line.startswith(':'):
      meta, path = line.split('\t', 1)
      blob = meta.split()[3]
      if blob.strip('0'):
        blobs[blob] = path
  if not blobs:
    return
  out = subprocess.run(['git', '-C', str(repo), 'cat-file', '--batch'], input=''.join(f'{b}\n' for b in blobs).encode(),
                       capture_output=True, check=True).stdout
  i = 0
  for path in blobs.values():
    end = out.index(b'\n', i)
    size = int(out[i:end].split()[2])
    yield path, out[end + 1:end + 1 + size]
    i = end + 1 + size + 1


# repo, branch, catalogs, source tag, clone filter, whether a catalog at a path lists only its big bundles
_HISTORY = (
  (SP_MODELS, 'gh-pages', 'docs/driving_models_*.json', 'sunnypilot-history', '--no-tags', _sunnypilot_big_only),
  (JETLINK, 'main', 'catalog/*.json', 'jetlink-history', '--filter=blob:none', lambda path: False),
)


def history(found: dict[str, Seen], work: Path) -> None:
  """Every catalog sunnypilot and jetlink ever published, from git history:
  a model dropped from today's catalog can still be on someone's comma."""
  for repo, branch, pathspec, tag, clone_filter, big_only in _HISTORY:
    dest = work / repo.replace('/', '_')
    subprocess.run(['git', 'clone', '-q', '--single-branch', '-b', branch, clone_filter, f'https://github.com/{repo}', str(dest)], check=True)
    for path, body in _git_blobs(dest, pathspec):
      try:
        data = json.loads(body)
      except ValueError:
        continue
      _catalog(found, _bundles(data, big_only(path)), f'{tag}:{path.rsplit("/", 1)[-1]}')


def jetlink(found: dict[str, Seen]) -> None:
  _catalog(found, _bundles(net.get_json(JETLINK_EXTRA), big_only=False), 'jetlink:extra_big_models.json')


def comma_master(found: dict[str, Seen], every_dir: bool = False) -> None:
  """Commits on master that changed the big ONNX or pkl. The pre-July-2026
  directory gets no new commits, so only a history run reads it."""
  pages = 50 if every_dir else 1
  for d in lfs.MODEL_DIRS if every_dir else lfs.MODEL_DIRS[:1]:
    for name in (lfs.BIG_ONNX, lfs.BIG_PKL):
      for c in net.github_pages(f'repos/{COMMA}/commits?sha=master&path={d}/{name}&per_page=100', max_pages=pages):
        _add(found, c['sha'], 'comma:master', c['commit']['message'].split('\n', 1)[0])


def _touches_model(number: str) -> bool:
  files = net.github_pages(f'repos/{COMMA}/pulls/{number}/files?per_page=100', max_pages=30)
  return any(f['filename'].endswith((lfs.BIG_ONNX, lfs.BIG_PKL)) for f in files)


def comma_prs(found: dict[str, Seen], checked: dict[str, list], pages: int = 1) -> None:
  """Heads of comma's pull requests that touch the big model. `checked` maps a
  PR number to [the head last looked at, whether it touched the model], so an
  unchanged PR costs nothing."""
  for pr in net.github_pages(f'repos/{COMMA}/pulls?state=all&sort=updated&direction=desc&per_page=100', max_pages=pages):
    number, head = str(pr['number']), pr['head']['sha']
    known = checked.get(number)
    if not known or known[0] != head:
      known = checked[number] = [head, _touches_model(number)]
    if known[1] and (seen := _add(found, head, 'comma:pull', f'#{number} {pr["title"]}')):
      seen.pr, seen.open = int(number), pr['state'] == 'open'


def huggingface_exports() -> dict[str, dict]:
  """oid -> {size, export} for every ONNX in comma's model repo."""
  exports = {}
  for e in net.get_json(HF_TREE):
    lfs_info = e.get('lfs') or {}
    if e.get('path', '').endswith('.onnx') and lfs_info.get('oid'):
      exports[lfs_info['oid']] = {'size': int(lfs_info['size']), 'export': e['path']}
  return exports
