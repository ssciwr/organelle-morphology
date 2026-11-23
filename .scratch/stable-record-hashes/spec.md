# Stable record filenames

Source: GitHub issue #7 — *record filenames use salted `hash()`, `analysis/` grows unboundedly across runs*.

## Problem

`Record.get_hash` (`src/organelle_morphology/records.py`) derives the on-disk
filename from Python's built-in `hash()`, which CPython salts per process via
`PYTHONHASHSEED`. The same logical record therefore gets a **new filename every
run**, `save_all_to_yaml` only writes (no pruning), and `load_all_from_yaml`
re-reads every historical generation.

A second, unnamed source of the same instability: `ProjectMetadata.blacklist`
and `whitelist` are built from **sets** (`project.py`), so their tuple order is
also salted per process. A stable digest must normalize that ordering or it
still drifts across processes.

## Design (agreed via grilling)

- **Identity-addressed** filename: `sha256` hex digest (64 chars) over the
  identity fields only — `name`, `meta`, `meta_project`, `meta_sources`.
  `data` is **excluded**, so a re-run with the same identity overwrites the same
  file in place (last-write-wins).
- **Canonicalization**: `dataclasses.asdict` → `Path`→`str` →
  `yaml.safe_dump(..., sort_keys=True)`. Sort `blacklist`/`whitelist` **at
  construction** so the metadata itself is canonical.
- **No pruning, no migration.** The loader **ignores old files** by loading only
  stems matching `^[0-9a-f]{64}$` (excludes old salted-int and legacy UUID stems).
- Keep **absolute paths** in identity. **Leave** legacy UUID-era files in place.
  `add()` / `Record.__eq__` / save logic stay untouched.

## Accepted boundaries (owned, not silent)

- Disk growth from **identity changes** (clipping / paths / source params) and
  **relocation** is *not* pruned — orphans accumulate. Only salt-induced
  duplication is fixed.
- **Pre-fix records are stranded** (ignored by the loader) until re-saved under
  the new scheme; import-without-rerun no longer works for old records.
- Source mutation under a fixed identity → silent overwrite (trusted invariant:
  data does not change without a metadata change).
- The `Record.__eq__` falsy-skipping walrus bug is out of scope (moot at load
  once old files are filtered).

## Acceptance criteria

- Same identity → **same filename** across `PYTHONHASHSEED=1` vs `2`, including
  with **non-empty** blacklist/whitelist.
- Changed `meta` → **different** filename.
- Re-saving the same identity → same filename, contents updated (idempotent
  overwrite); existing `test_records.py` overwrite assertions stay green.
- Loader skips old numeric / UUID stems and loads only new hex stems.
