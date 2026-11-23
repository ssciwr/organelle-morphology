# 03 — Records & schema: patch lists, versioning, clean break

Status: ready-for-agent
Spec: `../spec.md` · ADR: `docs/adr/0002-mcs-per-organelle-surface-distance.md`
Depends on: 01 (patch shape)

Multiple MCSs per pair require a record-schema break. Old results are
invalidated, not migrated (record dedup ignores data — `records.py:289–305` —
so without versioning, wrong areas would silently persist).

## Scope

- `Organelle._mcs[label][partner]` (`organelle.py:116,261–277`): list of patch
  entries `{patch_id, faces_index, distances, area, truncated,
  partner_patch_ids, meta}`; `add_mcs` appends, never overwrites.
- `McsData` (`organelle.py:73–89`): `n_contacts` = number of own patches
  across partners per label; add `n_partners`; total/mean/std_area over
  patches; distance mean area-weighted; `min_dist_per_org`/`max_dist_per_org`
  per partner (min/max over that partner's patches); normalizations unchanged.
- `calc_mcs_dict_entry` (:279–379): aggregate over the patch lists.
- `McsMetadata` (:40–59): add `schema_version` (current = 2); populate
  `filter_1`/`filter_2` from `mcs_meta` (today never set); take min/max from
  the label parameters, not the last loop entry (:320, :375–376).
- Record load (`records.py` `load_all_from_yaml`): reject records with
  missing/old `schema_version` — warn and list invalidated records.
- `generate_mcs` (`distance_calculations.py:559–680`): wire patch lists
  through `add_mcs`; keep `mcs_label` format and `project._mcs_labels`.
- Persistence stays aggregate-only (raw patch geometry in memory, as today).
  Per-patch persistence is out of scope.

## Acceptance

- One pair with two appositions → two patch entries per side, distinct
  `patch_id`s, correct per-patch areas; `n_contacts == 2`, `n_partners == 1`.
- U-case: sphere `n_contacts == 1`, U `n_contacts == 2` for the same label.
- Pre-fix YAML record (no schema_version) rejected on load with a warning;
  current-version record round-trips.
- `McsMetadata.filter_1/filter_2/min/max` correct for multi-partner organelles.

## Comments
