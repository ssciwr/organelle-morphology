# 04 — Analysis & UI: aggregation fixes, per-patch views

Status: resolved
Spec: `../spec.md` · ADR: `docs/adr/0002-mcs-per-organelle-surface-distance.md`
Depends on: 03

Consumers of McsData assume one record per organelle per label and one area
per partner. Update them for patch lists and fix the adjacent aggregation
bugs in the same code paths.

## Scope

- `Mcs_Analysis.get_mcs_properties` (`mcs_analysis.py:31–63`): keep one row
  per (label, organelle) aggregate; add `get_mcs_patches()` — in-memory
  per-patch table (organelle, partner, patch_id, area, distance stats,
  truncated).
- `get_mcs_overview` (:65–152): weight by patch counts; replace the
  partner-dedup that mutates `partners` lists (:96–107) with
  sets/frozensets.
- `analysis.get_contact_properties` (`analysis.py:126–142`): include new
  fields (`n_partners`, …); `get_contact_stats` (:220–242) overhauled for
  patch-based columns.
- `analysis.get_dataframe` (:244–306): replace `col.rsplit("-", 1)` parsing
  (:301) — mcs_labels contain `-`; use structured columns / MultiIndex.
- `Organelle.get_mesh_mcs_colored` (`organelle.py:197–217`): color per patch
  (tab20, i % 20), not per partner; scene/glb export inherits this.
- `src/app/ui.py`: verify MCS tables (:898, :913–914) and scene render with
  the new shape; surface `n_partners`/`truncated` where cheap. No structural
  redesign.

## Acceptance

- Overview/properties tables render for a multi-patch fixture; pair
  deduplication does not mutate record data.
- `get_dataframe` columns survive mcs_labels containing `-` (e.g. the
  default `0-0.1,mito-er` format).
- Scene colors distinguish two patches of the same partner.
- `tests/test_mcs_analysis.py` and `tests/test_analysis.py` updated to the
  patch-based semantics and passing.

## Comments

- 2026-11-23 — Resolved on integration branch `fix/mcs-area-fix` (merge `e09e1fb`; branch `fix/mcs-area-fix-04` @ `40755b6`). `Mcs_Analysis.get_mcs_patches()` returns an in-memory per-patch table (typed and empty after YAML-only load); `get_mcs_overview` weights by patch counts, dedups pairs with frozensets (no record mutation, no `list.index` failure) and reports `total_partners`/`mean_partners`. `get_contact_properties` includes `n_partners`, `min_dist`, `max_dist`; `get_contact_stats` returns `(mcs_label, property)` keys and `get_dataframe` returns a 2-level MultiIndex (`static` or the full MCS label × property), removing `rsplit("-", 1)` so labels such as `0-0.1,mito-er` survive. Scene coloring is per patch via `faces_index` (tab20). UI tables/plot formatting and an in-session patch table were adapted; no structural redesign. `tests/test_mcs_analysis.py`, `tests/test_analysis.py`, `tests/test_organelle.py` and `tests/test_source.py` updated and passing.
