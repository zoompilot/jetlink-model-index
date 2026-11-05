# jetlink-model-index

Which ONNX each big comma driving model is, worked out once an hour in CI so no car has to.

A jetlink server runs comma's big driving model from its ONNX. Catalogs name a model by the comma
commit it came from, and since openpilot #38930 many of those commits ship only a precompiled
tinygrad pkl: no ONNX in the tree, and often no word of which export it was built from
(recompiles, tinygrad bumps, reverts, force-pushed PR heads). Guessing from commit messages on
the device broke every time comma did something new. This repo answers the question here instead,
and every comma reads the answer.

## The index

`index/v1.json`, served from
`https://raw.githubusercontent.com/zoompilot/jetlink-model-index/main/index/v1.json`:

```json
{
  "schema": 1,
  "refs":   { "<comma commit>": { "oid": "<onnx sha256>", "size": 792305602, "method": "tree|weights|checkpoint|..." } },
  "onnx":   { "<onnx sha256>": { "size": 792305602, "export": "f22b8b34-.../16384/big_driving_supercombo.onnx" } },
  "models": [ { "ref": "...", "oid": "...", "size": 0, "name": "Light Speed Model v2 (October 29, 2026)", "folder": "2026 Deep RL Models",
                "catalog": "driving_models_chestnut_v28", "index": 19, "is_20hz": true, "minimum_selector_version": "21" } ],
  "unresolved": { "<comma commit>": "why" }
}
```

- `refs` is every big-model commit ever seen, in any catalog, on comma master or in a comma pull
  request. A commit stays here after a catalog drops it, so an old pick still resolves.
- `models` is what a Jetson can pick, in display order: **one entry per ONNX**, under the newest
  catalog entry that has it and in that entry's folder (Light Speed v1 and v2 are one ONNX, so one
  entry, "v2"). An open comma pull request whose ONNX no catalog lists yet goes in the
  `comma Pull Requests` folder, and drops out when the PR closes or sunnypilot catalogs it.
  `overrides.json` `folders` (`{"<ref>": "Folder"}`) moves any entry.
  Each entry has `ref`, `oid`, `size`, `name` and `folder`; one from a catalog also has that
  entry's `short_name`, `index`, `is_20hz`, `minimum_selector_version` and `build_time`, and one from
  a pull request its `pr`.
- The bytes behind an oid are on comma's LFS servers: the HuggingFace bucket `commaai/openpilot-lfs`
  (where comma's own LFS client stores objects since #39206), the older GitLab and HuggingFace
  `openpilot-lfs` LFS endpoints, and HuggingFace `openpilot_driving_models` for exports.

## How a commit is resolved

1. **override**: `overrides.json` says so. `{"refs": {"<ref>": {"oid": ..., "size": ...}}}`, or
   `{"unavailable": "why"}` for one that cannot be had, or `{"accept": "why"}` to keep a weaker
   answer that can never be checked (it stays out of the issue).
2. **tree**: the commit has `big_driving_supercombo.onnx` (under `openpilot/selfdrive/` or the
   pre-July-2026 `selfdrive/`). Its LFS pointer is the answer.
3. **weights**: the commit has only `big_driving_tinygrad.pkl`. The pkl carries its ONNX's
   initializers byte for byte, so it is matched against the library: a fingerprint of every ONNX
   comma has published (each HuggingFace export, and each ONNX any known commit has in its
   tree). comma's models are fine-tunes of each other and share most of their bytes, so a match
   is decided on the tensors only one ONNX has (`model_index/weights.py`).

   The pkl's metadata also names the export it was compiled from (`model_checkpoint`), read from
   its first 64 kB. The two must agree; the report flags any pkl where they do not.
4. **checkpoint**: the pkl's `model_checkpoint` alone, until a run has the download budget to
   check its weights.
5. **weights-sunnypilot**: comma's LFS servers no longer have the pkl (force-pushed PR heads,
   reverted commits), so sunnypilot's recompiled build of it is matched instead.
6. **message**: no bytes to check; the ONNX comma's commit or pull request diff deletes (which then
   joins the library), or the export their subjects name. The weakest; listed in the run's notes.

The first four are final. A ref resolved by checkpoint or message is tried again every run until
its weights can be checked.

A pkl that matches nothing is retried whenever the library grows. When comma's own commit
messages name an export it never uploaded, the report says so.

## Sources, checked every hour

- comma pull requests that touch the big model (their heads, before merge or force-push)
- comma master commits that change the big ONNX or pkl
- comma's HuggingFace model repo (`commaai/openpilot_driving_models`)
- every sunnypilot model catalog (`sunnypilot/sunnypilot-models`, gh-pages)
- jetlink's `catalog/extra_big_models.json` (resolved, no longer listed)

A commit's tree and messages never change, so each is read once and kept in `state.json`; which
catalogs and pull requests were already read is kept between runs in `.cache/` (actions/cache).
A quiet hour takes about 20 seconds.

A run with `history` also reads every version of every catalog either repo ever published.

## When something does not resolve

The run opens or updates the `unresolved` issue with what needs a human: commits nothing resolves,
pkls whose weights disagree with their own checkpoint, and refs still pending or answered only by
checkpoint or message two days after they turned up (a sign comma moved something the run reads). Fix one with an `overrides.json` entry;
the issue closes itself on the next clean run. Everything else worth a look (pending downloads,
checkpoint- or message-only answers, ties) goes to `notes.md` in the run's log.

## Running it

```bash
python -m model_index.build --history --work /path/with/2GB/free   # first run, or a rebuild
python -m model_index.build                                         # what the hourly job does
pytest -q && ruff check .
```

Stdlib only. Each download is deleted once fingerprinted or matched; only fingerprints
(`library/onnx/<oid>.json`, ~70 kB each) and results (`state.json`) are kept.
