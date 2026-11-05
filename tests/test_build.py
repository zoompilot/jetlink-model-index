"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink-model-index and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.
"""
from model_index import build

OID_A, OID_B = 'aa' * 32, 'bb' * 32


def test_export_names():
  # how comma names exports: the whole uuid, or its first eight and the checkpoint
  text = 'Update driving model to f22b8b34/16384 from 1a421175-db71-4e3d-9d62-e2166421b02b, not deadbeef'
  assert build._EXPORT.findall(text) == ['f22b8b34', '1a421175']


def test_settle_replaces_the_whole_result():
  entry = {'names': ['x'], 'status': 'resolved', 'oid': OID_A, 'method': 'message', 'ambiguous': [OID_A, OID_B]}
  build.settle(entry, 'unavailable', reason='never published')
  assert entry == {'names': ['x'], 'status': 'unavailable', 'reason': 'never published'}


def test_diff_pointer_finds_the_deleted_onnx():
  patch = '\n'.join([
    'From 1234 Mon Sep 17 00:00:00 2001', 'Subject: [PATCH] compiled', '',
    'diff --git a/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx b/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx',
    'deleted file mode 100644', '--- a/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx', '+++ /dev/null',
    '-version https://git-lfs.github.com/spec/v1', f'-oid sha256:{OID_A}', '-size 123',
  ])
  assert build.diff_pointer(patch).oid == OID_A
  assert build.patch_subjects(patch) == ['compiled']


def test_folder_in_either_overrides_shape():
  assert build._folder({'overrides': {'folder': 'Master Models'}}) == 'Master Models'
  assert build._folder({'overrides': [{'key': 'lat', 'value': '.1'}, {'key': 'folder', 'value': 'Master Models'}]}) == 'Master Models'
  assert build._folder({}) is None


def test_models_collapse_one_onnx_to_the_newest_entry(tmp_path, monkeypatch):
  monkeypatch.setattr(build, 'STATE', tmp_path / 'state.json')
  monkeypatch.setattr(build, 'LIBRARY', tmp_path / 'library')
  run = build.Run(tmp_path, 0, False)
  listing = lambda name, index, folder: {'display_name': name, 'index': index, 'overrides': {'folder': folder}}  # noqa: E731
  run.refs.update({
    'v1': {'status': 'resolved', 'oid': OID_A, 'size': 1, 'listing': {'driving_models_chestnut_v28': listing('Light Speed', 18, 'Deep RL')}},
    'v2': {'status': 'resolved', 'oid': OID_A, 'size': 1, 'listing': {'driving_models_chestnut_v28': listing('Light Speed v2', 19, 'Deep RL')}},
    'pr_same': {'status': 'resolved', 'oid': OID_A, 'size': 1, 'pr': 9, 'open': True, 'names': ['#9 Light speed']},
    'pr_new': {'status': 'resolved', 'oid': OID_B, 'size': 1, 'pr': 10, 'open': True, 'names': ['#10 Light speed 2']},
    'pr_closed': {'status': 'resolved', 'oid': 'cc' * 32, 'size': 1, 'pr': 8, 'open': False},
  })
  models = run.models(set(run.refs))
  assert [(m['ref'], m['name'], m['folder']) for m in models] == [('v2', 'Light Speed v2', 'Deep RL'), ('pr_new', '#10 Light speed 2', build.PR_FOLDER)]


def test_every_model_carries_what_a_catalog_needs_and_the_default_is_listed(tmp_path, monkeypatch):
  monkeypatch.setattr(build, 'STATE', tmp_path / 'state.json')
  monkeypatch.setattr(build, 'LIBRARY', tmp_path / 'library')
  monkeypatch.setattr(build, 'OVERRIDES', tmp_path / 'overrides.json')
  (tmp_path / 'overrides.json').write_text('{"default": "v1"}')
  run = build.Run(tmp_path, 0, False)
  listing = {'display_name': 'Light Speed v2', 'index': 19, 'is_20hz': True, 'generation': '12', 'environment': 'development',
             'runner': 'tinygrad', 'build_time': '2026-10-29T00:00:00Z', 'overrides': {'folder': 'Deep RL'}}
  run.refs.update({
    'v1': {'status': 'resolved', 'oid': OID_A, 'size': 1, 'method': 'tree'},
    'v2': {'status': 'resolved', 'oid': OID_A, 'size': 1, 'method': 'tree', 'listing': {'driving_models_chestnut_v28': listing}},
    'pr': {'status': 'resolved', 'oid': OID_B, 'size': 1, 'method': 'tree', 'pr': 10, 'open': True, 'names': ['#10 Light speed 2']},
  })
  index = run.index()
  v2, pr = index['models']
  assert (v2['short_name'], v2['generation'], v2['runner'], v2['build_time']) == ('v2', '12', 'tinygrad', '2026-10-29T00:00:00Z')
  assert (pr['short_name'], pr['generation'], pr['environment'], pr['is_20hz']) == ('#10', '12', 'development', True)
  # the default is named as the list names its ONNX
  assert index['default'] == 'v2'
  assert run.reports()[0] == ''


def test_unchecked_answers_reach_the_issue_after_two_days(tmp_path, monkeypatch):
  monkeypatch.setattr(build, 'STATE', tmp_path / 'state.json')
  monkeypatch.setattr(build, 'LIBRARY', tmp_path / 'library')
  monkeypatch.setattr(build, 'OVERRIDES', tmp_path / 'overrides.json')
  (tmp_path / 'overrides.json').write_text('{"refs": {"accepted": {"accept": "nothing else to go on"}}}')
  run = build.Run(tmp_path, 0, False)
  old, new = run.now - build.STALE - 1, run.now - 3600
  by_message = {'status': 'resolved', 'oid': OID_A, 'method': 'message', 'names': ['x']}
  run.refs.update({
    'fresh': {**by_message, 'since': new},
    'stale': {**by_message, 'since': old},
    'stale_pending': {'status': 'pending', 'reason': 'download budget spent', 'since': old},
    'stale_final': {'status': 'resolved', 'oid': OID_B, 'method': 'weights', 'since': old},
    'accepted': {**by_message, 'since': old},
  })
  human, info = run.reports()
  assert [line.split('`')[1] for line in human.splitlines()] == ['stale', 'stale_pending']
  assert [line.split('`')[1] for line in info.splitlines()] == ['accepted', 'fresh']
