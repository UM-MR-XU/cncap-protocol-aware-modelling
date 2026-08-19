# Clean-environment reproduction check

Run on 2026-08-19 in a container with **no access to the author's working tree**
and **no raw source data** — only this repository plus `pip install -r
requirements.txt`. This is the check that distinguishes "runs on my machine"
from "reproduces".

## Result

| Script | Tier | Status |
|---|---|---|
| `gbdt.py` (self-test) | A | ✅ ran to completion |
| `validators.py` | A | ✅ ran to completion |
| `dataset.py` | A | ✅ ran to completion |
| `cv_protocol.py` | A | ✅ ran to completion |
| `sigma_analysis.py` | A | ✅ ran to completion |
| `rule_layer.py` | A | ✅ ran to completion, output verified |
| `design_rules.py` | A | ✅ ran to completion, output verified |
| `monotonicity_check.py` | A | ✅ ran to completion |
| `baselines.py` | A | ✅ confirmed on a Windows workstation with LightGBM (8 s) |
| `oracle_zero.py` | A | ✅ after fixing the adjustments-table path (see change 7) |
| `ablation_e3.py` | A | ✅ confirmed on the same workstation (7 s) |
| `e5_min_config.py` | A | ✅ confirmed on the same workstation (5 s) |
| `build_long_table.py` | B | ✅ exits with guidance when `CNCAP_RAW_DIR` is unset |
| `validate_aggregation.py` | B | ✅ same |
| `patch_sigma.py` | B | ✅ same |

All fifteen scripts have now been confirmed end to end: the fast ones in a
clean Linux container without the author's working tree, the four model-training
ones on a Windows workstation with LightGBM installed. Total runtime for
`run_local.py` on that workstation was under one minute.

## Numbers regenerated in the clean environment

| Quantity | Regenerated | Paper |
|---|---|---|
| Rule-layer recovery | 99.6%, n=571 | 99.6%, n=571 |
| Per-revision complete layer | 100 / 100 / 100 / 99.1 / 99.0 / 100 / 100 % | identical |
| Thresholds-only aggregate | 94.2% | 94.2% |
| Design rules, indicator level | 17 / 31 / 40 | 17 / 31 / 40 |
| Design rules, configuration level | 19 / 21 / 14 | 19 / 21 / 14 |
| Modelled class, distinct indicators | 13 | 13 |
| Trivial baseline (always full marks), MAE | 0.1371 | 0.1371 |
| Trivial baseline (global mean), RMSE | 0.2454 | 0.2454 |
| Main model, matched distribution | MAE 0.0885, skill +0.3541 | +0.354 |

## Changes made for the public release

Three changes were needed to make the code run outside the author's machine.
They are recorded here because each one is a reproducibility failure mode that
would otherwise have shipped silently.

1. **`src/paths.py` located the project by walking up for a hard-coded folder
   name.** Replaced with a repo-relative root. Raw-data paths became optional,
   gated on `CNCAP_RAW_DIR`.
2. **Five scripts joined two raw scrape files to obtain official star ratings**
   (`rule_layer`, `design_rules`, `monotonicity_check`, `oracle_zero`, and
   `validate_aggregation`). Those files are not redistributable. The join result
   — de-identified — is now shipped as `data/official_outcomes.csv`, and the
   first four read it directly. `validate_aggregation` also needs the raw
   per-record JSON, so it stays in Tier B.
3. **Tier B scripts failed with `TypeError: ... not NoneType`** when the raw
   data was absent. They now exit with an explanatory message.
4. **`run_local.py` still checked for the raw scrape files during its
   environment self-check**, so the entry point aborted before running anything
   with `'NoneType' object has no attribute 'exists'`. It now checks the Tier A
   artefacts and reports whether Tier B is available.
5. **`run_local.py` ran `patch_sigma.py` and `build_long_table.py`
   unconditionally** as part of its data pipeline. Both are Tier B, so on a
   fresh clone the pipeline stopped at the first step. They are now skipped
   automatically when `CNCAP_RAW_DIR` is unset, because `data/` already contains
   what they would have produced.
6. **`run_local.py` ran `ablation_e3.py` before `design_rules.py`**, whose
   output it reads, and omitted `oracle_zero.py` and `e5_min_config.py`
   entirely — so the claim that a single entry point reproduces every result was
   not true for Tables 7 and 8. Order corrected, both scripts added.

Without change 1 the repository would not run anywhere. Without change 2 the
headline result (99.6%) would not have been reproducible by a reader.


## A note on the language of the code

Comments, configuration descriptions and some output labels are in Chinese,
because the analysis was developed in that language. The reader-facing
material — this file, `README.md`, `data/DATA_DICTIONARY.md` and `src/paths.py`
— is in English, and the output column *names* are ASCII. The class labels
inside `outputs/design_rules_*.csv` are still Chinese: `A 强准则` = strong rule,
`B 行业基线` = industry baseline, `C 需建模` = requires a model. Translating the
remainder is on the list; it does not affect reproducibility.


## Cross-implementation sensitivity found during this check

Running the suite under LightGBM, having previously run it under the
self-contained booster, changed three reported quantities enough to matter. Each
was corrected in the paper rather than left to a reader to discover.

| Quantity | Self-contained | LightGBM | Action |
|---|---|---|---|
| Marginal contribution, top three | 0.0554 / 0.0487 / 0.0466 | 0.0583 / 0.0482 / 0.0462 | unchanged claim: set and order both stable |
| Frontal airbag self-check | exactly 0.0000 | 0.0026 | paper now reports both; "exactly zero" is true only of the self-contained booster |
| Necessity rate, 2012 subset (n=31) | 96.8% / 90.3% | 87.1% / 84.9% | point values withdrawn; only the qualitative statement retained |

The first row is the reassuring one: the headline claim survives. The second and
third are the reason this check was worth running.

7. **`oracle_zero.py` read `test_level_adjustments.csv` from `outputs/`**, its
   location in the working tree, but the public release ships it in `data/`. It
   now checks both.
