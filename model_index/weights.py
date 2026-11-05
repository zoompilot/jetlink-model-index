"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink-model-index and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Which ONNX a precompiled tinygrad pkl was built from, by its weights.

Since openpilot #38930 a commit can ship only the compiled pkl. Its commit
message may or may not say which export it came from, and recompiles, tinygrad
bumps and reverts say nothing at all. The pkl does carry the ONNX's
initializers byte for byte, though, so the weights answer the question.

Comma's models are fine-tunes of each other: two exports share most of their
bytes (Light Speed and Train Track share 728 of 753 MB). What tells them apart
are the tensors only one of them has, so a match is judged on those.

A fingerprint keeps, per initializer of at least MIN_TENSOR bytes, a short hash
of the whole tensor (to find which tensors are shared between ONNX) and a
WINDOW-byte slice from its middle (to look for in a pkl). A few kB per ONNX
instead of 800 MB.
"""
from __future__ import annotations

import base64
import hashlib
import mmap
from collections import Counter
from pathlib import Path

MIN_TENSOR = 256
WINDOW = 64
QUICK = 12          # distinctive windows tried before the full check
FULL_MATCH = 0.98   # share of a candidate's windows a pkl must hold
# TensorProto fields that carry the bytes: raw_data, and the packed typed ones
_DATA_FIELDS = {9, 4, 5, 7, 10, 11}
_EXTERNAL = 14      # data_location; 1 is EXTERNAL


def _varint(b, i: int) -> tuple[int, int]:
  result = shift = 0
  while True:
    c = b[i]
    i += 1
    result |= (c & 0x7f) << shift
    shift += 7
    if c < 0x80:
      return result, i


def _fields(b, start: int, end: int):
  """(field, wire type, value or None, (start, end) or None) for a message."""
  i = start
  while i < end:
    key, i = _varint(b, i)
    number, wire = key >> 3, key & 7
    if wire == 0:
      value, i = _varint(b, i)
      yield number, wire, value, None
    elif wire == 1:
      yield number, wire, None, (i, i + 8)
      i += 8
    elif wire == 5:
      yield number, wire, None, (i, i + 4)
      i += 4
    elif wire == 2:
      n, i = _varint(b, i)
      yield number, wire, None, (i, i + n)
      i += n
    else:
      raise ValueError(f'unsupported wire type {wire} at {i}')


def initializers(b) -> list[tuple[str, int, int]]:
  """(name, start, end) of every initializer's bytes in an ONNX ModelProto."""
  found = []
  for number, wire, _, span in _fields(b, 0, len(b)):
    if number != 7 or wire != 2:  # ModelProto.graph
      continue
    for g, gwire, _, gspan in _fields(b, *span):
      if g != 5 or gwire != 2:  # GraphProto.initializer
        continue
      name, data, external = '', None, False
      for t, twire, value, tspan in _fields(b, *gspan):
        if t == 8 and twire == 2:
          name = bytes(b[tspan[0]:tspan[1]]).decode('utf-8', 'replace')
        elif t in _DATA_FIELDS and twire == 2:
          data = tspan
        elif t == _EXTERNAL and value == 1:
          external = True
      if data is not None and not external:
        found.append((name, *data))
  return found


def _window(b, start: int, end: int) -> bytes:
  mid = start + (((end - start) // 2) // WINDOW) * WINDOW
  return bytes(b[mid:mid + WINDOW])


def fingerprint(path: Path) -> list[list]:
  """[[tensor hash, bytes, base64 window], ...] for an ONNX file."""
  with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
    tensors = []
    for _, start, end in initializers(b):
      if end - start < MIN_TENSOR:
        continue
      digest = hashlib.sha256(memoryview(b)[start:end]).hexdigest()[:16]
      tensors.append([digest, end - start, base64.b64encode(_window(b, start, end)).decode()])
  if not tensors:
    raise ValueError(f'{path} has no initializers of {MIN_TENSOR} bytes or more')
  return tensors


def _spread(items: list, n: int) -> list:
  if len(items) <= n:
    return items
  return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def match(pkl: Path, library: dict[str, list[list]]) -> dict[str, list]:
  """{oid: [share of windows found, has tensors of its own]} for every ONNX in
  `library` (oid -> fingerprint) whose weights `pkl` carries.

  Each candidate is first tried on QUICK of its distinctive windows, ones from
  tensors no other ONNX in the library has, and stops at the first miss, so a
  wrong one costs a scan or two. One that passes is checked on every window,
  and stops once it has missed more than FULL_MATCH allows. A window is looked
  for once per pkl however many candidates share it, starting where the last
  one was found, since a pkl keeps an ONNX's tensors mostly in order."""
  counts = Counter(d for tensors in library.values() for d in {t[0] for t in tensors})
  with open(pkl, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as b:
    seen: dict[str, bool] = {}
    last = 0

    def found(window: str) -> bool:
      nonlocal last
      if window not in seen:
        w = base64.b64decode(window)
        at = b.find(w, last)
        if at < 0:
          at = b.find(w, 0, last + WINDOW)
        if at >= 0:
          last = at
        seen[window] = at >= 0
      return seen[window]

    full = {}
    for oid in sorted(library):
      tensors = library[oid]
      distinct = [t for t in tensors if counts[t[0]] == 1]
      if not all(found(t[2]) for t in _spread(distinct or tensors, QUICK)):
        continue
      allowed, misses = int((1 - FULL_MATCH) * len(tensors)), 0
      for t in tensors:
        misses += not found(t[2])
        if misses > allowed:
          break
      else:
        full[oid] = [round(1 - misses / len(tensors), 4), bool(distinct)]
  return full


def pick(full: dict[str, list], priors: list[str] = ()) -> tuple[str | None, list[str]]:
  """The ONNX a pkl was built from, out of match()'s full matches, and the
  candidates it could not choose between. One with tensors of its own beats
  ones that only share (re-exports with the same weights); among those left,
  the first of `priors` wins."""
  best = sorted(o for o in full if full[o][1]) or sorted(full)
  if len(best) <= 1:
    return (best[0] if best else None), []
  for p in priors:
    if p in best:
      return p, []
  return best[0], best
