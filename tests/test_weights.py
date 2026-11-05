"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink-model-index and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.
"""
import os
import random

import pytest

from model_index import weights


def _varint(n: int) -> bytes:
  out = bytearray()
  while True:
    b = n & 0x7f
    n >>= 7
    out.append(b | (0x80 if n else 0))
    if not n:
      return bytes(out)


def _field(number: int, payload: bytes) -> bytes:
  return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def _onnx(tensors: dict[str, bytes]) -> bytes:
  """A ModelProto with only initializers: graph(7) -> initializer(5) -> name(8), data_type(2), raw_data(9)."""
  inits = b''.join(_field(5, _field(8, name.encode()) + _varint(2 << 3) + _varint(10) + _field(9, raw)) for name, raw in tensors.items())
  return _varint(1 << 3) + _varint(9) + _field(7, _field(1, b'node') + inits)


def _blob(seed: int, n: int = 4096) -> bytes:
  return random.Random(seed).randbytes(n)


@pytest.fixture
def models(tmp_path):
  """Two fine-tunes sharing a backbone, a re-export of the first, and a pkl
  built from the first."""
  backbone = {f'b{i}': _blob(i) for i in range(20)}
  a = {**backbone, 'head_a': _blob(100), 'head_a2': _blob(101)}
  b = {**backbone, 'head_b': _blob(200), 'head_b2': _blob(201)}
  paths = {}
  for name, tensors in (('a', a), ('b', b)):
    paths[name] = tmp_path / f'{name}.onnx'
    paths[name].write_bytes(_onnx(tensors))
  # a pkl: pickle framing around the same bytes, in another order, with padding
  pkl = tmp_path / 'a.pkl'
  pkl.write_bytes(b'\x80\x05' + os.urandom(999) + b''.join(raw + b'\0' * 7 for raw in reversed(list(a.values()))))
  return paths, pkl


def test_initializers_found(models):
  paths, _ = models
  assert len(weights.fingerprint(paths['a'])) == 22


def _library(paths, **extra):
  return {'a': weights.fingerprint(paths['a']), 'b': weights.fingerprint(paths['b']), **extra}


def test_shared_backbone_does_not_fool_the_match(models):
  paths, pkl = models
  full = weights.match(pkl, _library(paths))
  assert set(full) == {'a'} and full['a'][1]
  assert weights.pick(full) == ('a', [])


def test_prior_does_not_override_the_weights(models):
  paths, pkl = models
  assert weights.pick(weights.match(pkl, _library(paths)), priors=['b']) == ('a', [])


def test_unknown_onnx_matches_nothing(models):
  paths, pkl = models
  assert weights.match(pkl, {'b': weights.fingerprint(paths['b'])}) == {}
  assert weights.pick({}) == (None, [])


def test_reexport_with_same_weights_ties_until_a_prior_breaks_it(models):
  paths, pkl = models
  fp = weights.fingerprint(paths['a'])
  full = weights.match(pkl, {'a1': fp, 'a2': fp, 'b': weights.fingerprint(paths['b'])})
  assert set(full) == {'a1', 'a2'}
  assert weights.pick(full) == ('a1', ['a1', 'a2'])
  assert weights.pick(full, priors=['x', 'a2']) == ('a2', [])


def test_external_data_is_skipped(tmp_path):
  raw = _blob(1)
  external = _field(5, _field(8, b'x') + _varint(14 << 3) + _varint(1) + _field(9, raw))
  path = tmp_path / 'ext.onnx'
  path.write_bytes(_field(7, external + _field(5, _field(8, b'y') + _field(9, _blob(2)))))
  assert [name for name, *_ in weights.initializers(path.read_bytes())] == ['y']
