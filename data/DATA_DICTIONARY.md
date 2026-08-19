# Data dictionary

All tables are de-identified. Records are keyed by an opaque `variant_id`;
vehicle model names, manufacturer names and source URLs have been removed.

Revision codes: `V1`=2006, `V2`=2009, `V3`=2012, `V4`=2015, `V5`=2018,
`V6`=2021, `V7`=2024.

---

## `train_long.csv.gz` — indicator-level long table (13,060 rows)

One row per (vehicle, indicator). Structural absence is represented by the
absence of a row, so no masking is needed.

| Column | Meaning |
|---|---|
| `variant_id` | opaque vehicle identifier |
| `version` | protocol revision code |
| `test_year` | year of test |
| `item_id` | `occ.{scenario}.{position}.{region}` |
| `item_id_coarse` | identifier with merged reporting variants |
| `scenario` | test scenario, e.g. `frontal_frb100`, `side_mdb` |
| `dummy` | seating position group: front row, second-row adult, second-row child |
| `region` | body region, e.g. `chest`, `leg`, `headneck` |
| `role` | `target` (regression target, 11,065 rows), `aggregation_source` (996), `rule_unit_test` (996), `penalty` (3) |
| `observability_class` | `config_gated`, `structure_driven`, `component_driven` |
| `score` | official indicator score |
| `max` | maximum attainable for that indicator under that revision |
| `y` | normalised score, `score / max`, in [0, 1] |
| `sigma_speed` | impact speed or speed change |
| `sigma_barrier` | barrier type |
| `sigma_dummy` | dummy type |

The three `sigma_*` columns carry three distinct states: a measured value; an
explicit not-applicable marker for non-impact evaluation items; and absence,
meaning the scenario does not exist in that revision. Collapsing these three
into one would convert "not applicable" into a specific wrong value.

**Only rows with `role == 'target'` are used to fit models.** The pipeline
asserts this rather than assuming it.

---

## `vehicle_features.csv` — design features, one row per vehicle

Gross parameters (dimensions, masses, displacement, price), passive safety
configuration (airbags by type and location, belt pretensioners and load
limiters, child seat anchorages, seat occupancy monitoring) and component-level
features where available.

Which columns may enter the model is declared in `config/feature_groups.json`,
not in code. Active safety configuration is present in the file but excluded
from modelling: those systems act before the collision and have no causal
pathway to post-impact dummy loading.

---

## `official_outcomes.csv` — official result per vehicle (571 rows)

| Column | Meaning |
|---|---|
| `variant_id` | opaque vehicle identifier |
| `version` | protocol revision code |
| `star_rating` | official star rating |
| `overall_score` | official overall score or score rate |
| `ap_score` | occupant protection module rate |
| `vru_score` | vulnerable road user module rate |
| `as_score` | active safety module rate |
| `test_year` | year of test |

This is the target against which the rule layer is validated. It contains 571
vehicles; the modelling tables contain 570, the difference being one record
whose indicator-level breakdown is corrupt but whose test-item level is intact.

---

## `test_level_adjustments.csv` — non-additive adjustments

Difference between the official test-item score and the sum of its indicators,
classified as penalty (162), bonus (24), conditional zeroing (2) or none
(1,969), over 2,157 vehicle–test combinations.

These values appear only in free-text annotation on the source pages. They can
be recovered by inversion during training but are unavailable when predicting an
untested vehicle, which is why results are reported in both an oracle mode and a
zero mode.
