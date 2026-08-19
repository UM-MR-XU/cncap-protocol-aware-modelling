# Protocol-Aware Modelling of Evolving Assessment Systems

Code and data for the paper *"Protocol-Aware Modelling of Evolving Assessment
Systems: A Three-Layer Transferable Decomposition"* (IEEE Access, under review).

Archived release: <https://doi.org/10.5281/zenodo.22014824>

Standardised assessment protocols are revised on a cycle, and each revision
changes **what is measured**, not merely how strictly it is judged. A model
trained on earlier revisions has no output for an indicator a later revision
introduces, so it cannot run at all. This repository implements the three-layer
decomposition proposed to address that: a **physical layer** that is learned, a
**rule layer** that is loaded from the published protocol rather than fitted,
and an **assumption layer** whose priors are ablated rather than assumed.

---

## 1. Quick start

```bash
git clone https://github.com/UM-MR-XU/cncap-protocol-aware-modelling.git
cd cncap-protocol-aware-modelling
pip install -r requirements.txt
python run_local.py
```

Nothing else needs to be downloaded. The de-identified derivative dataset is
included in `data/`.

`run_local.py` runs every step in the correct order, including the two
long-running experiments behind Tables 7 and 8.

Expected runtime: under a minute on a workstation with LightGBM installed;
substantially longer without it, because the fallback is the self-contained
pure-NumPy booster in `src/gbdt.py`.

Only one ordering constraint exists if you run scripts individually:
`design_rules.py` must precede `ablation_e3.py`, which reads its output.

---

## 2. What reproduces what

Every number in the paper maps to a script and an output file.

| Paper location | Command | Output |
|---|---|---|
| Table 3, rule layer recovery 99.6% | `python src/rule_layer.py` | `outputs/rule_layer_validation.json` |
| Sec. V-A, discriminability bound (11 of 35) | read from `config/rule_config.json` `_provenance` | n/a |
| Table 4, 2x2 factorial counts | `python src/sigma_analysis.py` | `outputs/sigma_analysis.json` |
| Table 5, skill scores under both implementations | `python run_local.py` | `outputs/baseline_forward_transfer__npy.csv`, `__lgb.csv` |
| Table 6, controlled contrasts 5.8x / 6.6x | same as Table 5 | same |
| Sec. V-C, design rules 17/31/40 | `python src/design_rules.py` | `outputs/design_rules_items.csv`, `design_rules_report.md` |
| Sec. V-D, monotonicity ablation (negative result) | `python src/ablation_e3.py` | `outputs/ablation_e3__npy.csv` |
| Sec. V-D, the five monotonicity checks | `python src/monotonicity_check.py` | `outputs/monotonicity_check.json` |
| Table 7, oracle / zero dual mode | `python src/oracle_zero.py` | `outputs/oracle_zero.csv`, `oracle_zero_star.csv` |
| Sec. V-E, observability strata | same as Table 7 | `outputs/e4_observability.csv` |
| Table 8, configuration necessity under both weightings | `python src/e5_min_config.py` | `outputs/e5_min_config.csv` |
| Sec. IV, dataset composition and invariants | `python src/validators.py` | `outputs/validation_report.json` |

A snapshot of all of these is committed under `outputs/`, so the reported
numbers can be checked without running anything.

### Which backend the committed snapshot uses

Four artefacts depend on which gradient booster is installed, and are therefore
committed under both backends:

| File | Backend |
|---|---|
| `*.csv` (no suffix) | self-contained booster. **These are the values in the paper's tables.** |
| `*__npy.csv` | identical copy, explicitly labelled |
| `*__lgb.csv` | LightGBM |

This applies to `baseline_group_cv`, `baseline_forward_transfer`, `ablation_e3`
and `e5_min_config`. Running with LightGBM installed will therefore not
reproduce the unsuffixed files byte for byte; compare against `*__lgb.csv`
instead. The differences are small but real, and Section V of the paper reports
where they matter. Most notably, the frontal-airbag self-check returns exactly
0.0000 only under the self-contained booster; LightGBM returns 0.0026.

---

## 3. Two tiers of reproduction

**Tier A, the default.** Everything above runs from `data/`, which contains the
de-identified derivative dataset. No configuration required.

**Tier B, optional.** `build_long_table.py`, `validate_aggregation.py` and
`patch_sigma.py` rebuild the dataset from the raw assessment records. Those
records are published by the protocol authority but are **not redistributed
here** (see section 5). To enable this tier, point an environment variable at a
local copy:

```bash
export CNCAP_RAW_DIR=/path/to/C-NCAP_Test_Data     # macOS / Linux
set CNCAP_RAW_DIR=C:\path\to\C-NCAP_Test_Data      # Windows
```

Without it, these three scripts exit with an explanatory message rather than a
stack trace.

---

## 4. Repository layout

```
run_local.py              one-command entry point
requirements.txt
src/                      16 scripts, all independently runnable
    paths.py              all paths in one place; repo-relative
    gbdt.py               self-contained gradient booster (pure NumPy)
    rule_layer.py         protocol rules -> star rating
    dataset.py            design matrix assembly + leakage self-check
    cv_protocol.py        grouped CV and rolling-origin folds
    baselines.py          B0 / B1 / B2
    design_rules.py       deterministic design-rule mining
    ablation_e3.py        monotonicity ablation
    oracle_zero.py        oracle/zero dual mode + observability strata
    e5_min_config.py      per-item counterfactual ablation
config/
    rule_config.json      every rule parameter, with _provenance
    feature_groups.json   which features may enter the model, and why
    supplement_v6_side_mdb_row2.json
ontology/
    item_ontology_v1.json 62 indicator identifiers + full sigma table
data/                     de-identified derivative, see section 5
    train_long.csv.gz     13,060 rows
    vehicle_features.csv
    test_level_adjustments.csv
    official_outcomes.csv official star and module rates per variant
    DATA_DICTIONARY.md
outputs/                  committed snapshot of all reported results
```

Two files are worth reading even if you never run the code:

- **`config/rule_config.json`** carries a `_provenance` block recording, for
  every rule parameter, which edition of the protocol it was read from and
  whether the dataset can falsify it. This is the artefact behind the
  discriminability analysis.
- **`config/feature_groups.json`** declares which features are admitted to the
  model. Active safety configuration is excluded there, not in code, so the
  exclusion is auditable rather than asserted after the fact.

---

## 5. Data provenance

The underlying assessment results and technical protocols are **published by
the protocol authority** and were retrieved from its public pages.

This repository ships a **structured derivative**, not the raw records:

- vehicle model names, manufacturer names and source URLs have been removed;
  each record is identified only by an opaque `variant_id`
- the derivative is the long-format indicator table plus the feature, adjustment
  and official-outcome tables that the analysis consumes
- the raw scrape and the protocol documents are not redistributed, because their
  licensing is not ours to grant

The derivative is sufficient to reproduce every result in the paper.

---

## 6. Known limitations

These are stated in the paper and repeated here so that anyone running the code
is not surprised by them.

- **One row was excluded in error.** A body-region name was mistranscribed in a
  single record; the correct action would have been to rename it rather than
  drop it. The row is still excluded, because re-running would shift every
  published count with no change to any conclusion. See the paper, Sec. IV-C.
- **One record is excluded in full** (`V7/12`). Its page parse collapsed:
  module-level summaries appear inside the test-item list, a pedestrian
  protection item is broadcast across all six test blocks, and the occupant
  protection score rate appears twice with conflicting values. Consequently the
  rule layer is validated on 571 vehicles while the model is fitted on 570.
- **The most recent revision contributes 14 vehicles**, and all five of its star
  thresholds are non-falsifiable with this data. Conclusions involving it are
  indicative only.
- **The scoring curve is not separated from the physical layer**, because injury
  readings are not published. This is the framework's main limitation and is
  quantified in the paper rather than asserted away.
- **No causal claim is made.** The counterfactual ablation is inference within a
  fitted model, not an estimate of a causal effect.

---

## 7. Environment

Developed and tested on Python 3.11. `src/gbdt.py` is a self-contained
implementation requiring only NumPy, so the cross-implementation check runs even
where LightGBM is unavailable.

A note for Windows users editing these files: `Set-Content` and `>` in
PowerShell write in the system code page, which corrupts UTF-8. Use
`Set-Content -Encoding utf8` or an editor that preserves the encoding.

---

## 8. Citation

See `CITATION.cff`. Please cite the paper rather than this repository alone.

## 9. Licence

Code is released under the MIT Licence (`LICENSE`). The derivative data in
`data/` and `outputs/` is released under CC BY 4.0 (`LICENSE-data`).
