"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink-model-index and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The big model's files at a comma commit, and the bytes behind them.

GitHub's raw host serves a file for any commit it still holds, merged or not,
including a pull request head that was force-pushed away. Most are git-lfs
pointers whose bytes are on comma's LFS servers; which server has an object
depends on its age, so each is asked in turn. Pkls split into chunks (before
openpilot #38941) were committed as plain git files instead.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from model_index import net

# where the big model has lived: the tree moved under openpilot/ in July 2026
MODEL_DIRS = ('openpilot/selfdrive/modeld/models', 'selfdrive/modeld/models')
BIG_ONNX = 'big_driving_supercombo.onnx'
BIG_PKL = 'big_driving_tinygrad.pkl'
RAW_URL = 'https://raw.githubusercontent.com/commaai/openpilot/{ref}/{path}'
DRIVING_MODELS_REPO = 'commaai/openpilot_driving_models'
ENDPOINTS = (
  'https://gitlab.com/commaai/openpilot-lfs.git/info/lfs',
  'https://huggingface.co/commaai/openpilot-lfs.git/info/lfs',
  f'https://huggingface.co/{DRIVING_MODELS_REPO}.git/info/lfs',
)
POINTER_MAX = 4096
CHUNK = 8 << 20
MEDIA = 'application/vnd.git-lfs+json'
SHA256 = re.compile(r'[0-9a-f]{64}')


@dataclass(frozen=True)
class Pointer:
  """An object by its hash: an LFS object (sha256, asked of ENDPOINTS), or at
  `url`, a plain git blob (sha1) or a plain file (sha256, size 0 if unknown)."""
  oid: str
  size: int
  url: str | None = None


def parse_pointer(text: str) -> Pointer | None:
  if len(text) > POINTER_MAX or not text.startswith('version https://git-lfs'):
    return None
  oid = size = None
  for line in text.splitlines():
    key, _, value = line.partition(' ')
    if key == 'oid':
      oid = value.removeprefix('sha256:').strip()
    elif key == 'size' and value.strip().isdigit():
      size = int(value)
  if oid is None or not SHA256.fullmatch(oid) or not size:
    return None
  return Pointer(oid, size)


def _fetch_pointer(url: str) -> Pointer:
  pointer = parse_pointer(net.get(url, limit=POINTER_MAX + 1).decode('utf-8', 'replace'))
  if pointer is None:
    raise net.NetworkError(f'{url} is not an lfs pointer')
  return pointer


def pointer_at(ref: str, name: str) -> tuple[Pointer, str] | None:
  """The pointer for `name` in the model directory at `ref`, and its path, or
  None when the commit does not have that file."""
  for d in MODEL_DIRS:
    try:
      return _fetch_pointer(RAW_URL.format(ref=ref, path=f'{d}/{name}')), f'{d}/{name}'
    except net.NotFound:
      continue
  return None


_CHUNK = re.compile(re.escape(BIG_PKL) + r'\.chunk(\d+)of(\d+)$')


def pkl_at(ref: str) -> list[Pointer] | None:
  """The big pkl at `ref`: one pointer, or its chunks in order. None when the
  commit has neither."""
  found = pointer_at(ref, BIG_PKL)
  if found is not None:
    return [found[0]]
  for d in MODEL_DIRS:
    try:
      listing = net.github(f'repos/commaai/openpilot/contents/{d}?ref={ref}')
    except net.NotFound:
      continue
    chunks = sorted((int(m.group(1)), int(m.group(2)), e) for e in listing if (m := _CHUNK.match(e.get('name', ''))))
    if not chunks:
      continue
    total = chunks[0][1]
    if [c[:2] for c in chunks] != [(i, total) for i in range(1, total + 1)]:
      raise net.NetworkError(f'{ref[:10]}: {len(chunks)} pkl chunks do not make a set of {total}')
    pointers = []
    for _, _, e in chunks:
      url = RAW_URL.format(ref=ref, path=f'{d}/{e["name"]}')
      # a chunk small enough to be a pointer is one; the rest are the bytes themselves
      pointers.append(_fetch_pointer(url) if e['size'] <= POINTER_MAX else Pointer(e['sha'], int(e['size']), url))
    return pointers
  return None


def pkl_key(pointers: list[Pointer]) -> str:
  """One name for a pkl, whole or chunked: its oid, or a hash of its chunks' oids."""
  if len(pointers) == 1:
    return pointers[0].oid
  return 'chunks:' + hashlib.sha256(''.join(p.oid for p in pointers).encode()).hexdigest()


def hrefs(pointer: Pointer):
  """(endpoint, href) for each LFS server that has the object."""
  body = json.dumps({'operation': 'download', 'transfers': ['basic'],
                     'objects': [{'oid': pointer.oid, 'size': pointer.size}]}).encode()
  for endpoint in ENDPOINTS:
    try:
      payload = net.get_json(f'{endpoint}/objects/batch', data=body, headers={'Accept': MEDIA, 'Content-Type': MEDIA})
    except (net.NetworkError, net.NotFound):
      continue
    for obj in (payload.get('objects') or []) if isinstance(payload, dict) else []:
      if not isinstance(obj, dict) or obj.get('oid') != pointer.oid or 'error' in obj:
        continue
      if href := ((obj.get('actions') or {}).get('download') or {}).get('href'):
        yield endpoint, href


def _sources(pointer: Pointer):
  if pointer.url:
    yield pointer.url, pointer.url
  else:
    yield from hrefs(pointer)


def head(pointer: Pointer, n: int = 1 << 16) -> bytes:
  """The first `n` bytes of an object, by a range request."""
  rng = {'Range': f'bytes=0-{n - 1}'}
  for _, url in _sources(pointer):
    try:
      return net.get(url, headers=rng, limit=n)
    except (net.NetworkError, net.NotFound):
      continue
  raise net.NetworkError(f'no server would serve the head of {pointer.oid[:16]}')


def _hasher(pointer: Pointer):
  # a git blob's oid is the sha1 of a header and its bytes
  return hashlib.sha1(f'blob {pointer.size}\0'.encode()) if len(pointer.oid) == 40 else hashlib.sha256()


def _fetch(pointer: Pointer, out) -> None:
  """Append an object's bytes to `out`, from the first source that delivers
  the right ones."""
  start, errors = out.tell(), []
  for source, url in _sources(pointer):
    digest, n = _hasher(pointer), 0
    try:
      with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': net.USER_AGENT}), timeout=120) as r:
        while chunk := r.read(CHUNK):
          out.write(chunk)
          digest.update(chunk)
          n += len(chunk)
      if (not pointer.size or n == pointer.size) and digest.hexdigest() == pointer.oid:
        return
      errors.append(f'{source[:60]}: wrong bytes')
    except OSError as e:
      errors.append(f'{source[:60]}: {e}')
    out.seek(start)
    out.truncate()
  raise net.NetworkError(f'no source delivered {pointer.oid[:16]}: {"; ".join(errors) or "none has it"}')


def download(pointers: list[Pointer], dest: Path) -> Path:
  """An object, or a pkl's chunks joined in order, at `dest`."""
  with open(dest, 'wb') as out:
    for p in pointers:
      _fetch(p, out)
  return dest


_CHECKPOINT = re.compile(rb'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/(\d+)')


def checkpoint(pkl_head: bytes) -> list[str]:
  """The '<uuid>/<step>' pairs of the model_checkpoint comma's compiler writes
  into a pkl's metadata, the export the pkl was built from last. Empty when
  the pkl has none."""
  i = pkl_head.find(b'model_checkpoint')
  if i < 0:
    return []
  return [f'{u.decode()}/{s.decode()}' for u, s in _CHECKPOINT.findall(pkl_head[i:i + 512])[:4]]
