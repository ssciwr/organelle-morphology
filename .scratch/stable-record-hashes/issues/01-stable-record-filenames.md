# 01 — Stable record filenames + ignore old files

Type: task
Status: ready-for-agent
Blocked by: —

Spec: `.scratch/stable-record-hashes/spec.md` (source: GitHub issue #7).

## What

Replace the salted `hash()` filename with a stable, process-independent sha256
identity digest, normalize set-derived metadata ordering, and make the loader
ignore pre-fix files. No pruning, no migration.

## Changes

1. `src/organelle_morphology/records.py`
   - `Record.get_hash()` → return a 64-char lowercase sha256 hex `str` (was a
     salted signed `int`). Digest covers `name`, `meta`, `meta_project`,
     `meta_sources` only (NOT `data`).
   - Add a canonicalization helper: `dataclasses.asdict` → `Path`→`str` →
     `yaml.safe_dump(..., sort_keys=True)`.
   - `RecordRegistry.load_all_from_yaml`: filter `rglob("*.yaml")` results to
     stems matching `^[0-9a-f]{64}$`.
   - Sole caller `save_all_to_yaml` (`records.py:405`) is an f-string; sidecar
     `<stem>_<field>.npz` names follow the new stem automatically.
2. `src/organelle_morphology/project.py` (`metadata` property, ~232-233)
   - `tuple(sorted(self.permanent_blacklist))` / `tuple(sorted(self.permanent_whitelist))`.
3. Short user-facing doc note: filenames are now stable; pre-fix files are
   ignored by the loader.

## Tests (pytest; `project_with_sources` + cheap `ProfileCalculator` pipeline)

- Cross-process: two subprocesses with `PYTHONHASHSEED=1` vs `2`, non-empty
  blacklist/whitelist → identical filename. (New subprocess pattern for suite.)
- Changed `meta` → different filename.
- Same identity re-saved → same filename, contents updated.
- Loader ignores old numeric / UUID stems, loads only 64-hex stems.
- Keep existing `test_records.py` overwrite assertions green. Windows CI safe.

## Acceptance

See spec. Run `python -m pytest` (conda env) — all green.

## Comments

- Design settled via grilling; identity-addressed (B), no pruning (C), ignore
  old files via loader stem-filter (chosen over migration). Absolute paths kept.
- Implemented on branch `fix/stable-record-hashes` (worktree
  `worktrees/stable-record-hashes`). `get_hash` → sha256 hex + `_canonical_identity`;
  loader stem-filter; `project.py` sorts blacklist/whitelist; short usage.md note;
  6 new tests in `tests/test_records.py`.
- Verified (conda env `morph`, py3.12): full suite **186 passed**; ruff check +
  format clean. Cross-process `PYTHONHASHSEED=1` vs `2` test confirms stability.
  Not yet committed.
