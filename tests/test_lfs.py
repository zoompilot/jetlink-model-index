"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink-model-index and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.
"""
from model_index import lfs

POINTER = 'version https://git-lfs.github.com/spec/v1\noid sha256:' + 'ab' * 32 + '\nsize 790445184\n'


def test_parse_pointer():
  assert lfs.parse_pointer(POINTER) == lfs.Pointer('ab' * 32, 790445184)


def test_parse_pointer_rejects_other_text():
  assert lfs.parse_pointer('<html>') is None
  assert lfs.parse_pointer(POINTER.replace('ab' * 32, 'zz')) is None


def test_checkpoint():
  head = b'\x80\x05}\x94\x8c\x10model_checkpoint\x94\x8cUf22e06e5-3b3f-48d2-a6b9-2f32dc20e707/56320/f22b8b34-0983-4185-a453-a87b73798d61/16384\x94u'
  assert lfs.checkpoint(head) == ['f22e06e5-3b3f-48d2-a6b9-2f32dc20e707/56320', 'f22b8b34-0983-4185-a453-a87b73798d61/16384']
  assert lfs.checkpoint(b'no metadata') == []
