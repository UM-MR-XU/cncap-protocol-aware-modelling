# -*- coding: utf-8 -*-
"""
Path configuration for the public release.

Two tiers of reproduction are supported.

  Tier A (default, no extra downloads)
      Everything downstream of the long table runs from the de-identified
      derivative shipped in data/.  This covers every number reported in the
      paper except the construction of the long table itself.

  Tier B (optional, requires the raw source records)
      build_long_table.py regenerates data/train_long.csv.gz from the raw
      assessment records.  Those records are published by the protocol
      authority but are not redistributed here; see README, "Data provenance".
      Point CNCAP_RAW_DIR at a local copy to enable this tier:

          export CNCAP_RAW_DIR=/path/to/C-NCAP_Test_Data     # macOS / Linux
          set CNCAP_RAW_DIR=C:\\path\\to\\C-NCAP_Test_Data    # Windows

The original working-tree version of this file located the project by walking
up for a hard-coded folder name.  That is exactly the kind of dependency that
makes a repository run only on its author's machine, so it has been replaced by
a repo-relative root.
"""
import os
from pathlib import Path

# ── repository root: the parent of src/ ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
ONTO_DIR = ROOT / "ontology"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ── ontology and configuration ───────────────────────────────────────────────
ONTO = ONTO_DIR / "item_ontology_v1.json"
MAPPING = ONTO_DIR / "item_mapping.csv"          # optional, not required to run
RULE_CONFIG = CONFIG_DIR / "rule_config.json"
FEATURE_GROUPS = CONFIG_DIR / "feature_groups.json"
SUPPLEMENT = CONFIG_DIR / "supplement_v6_side_mdb_row2.json"

# ── Tier A artefacts: shipped, read by everything downstream ─────────────────
LONG_TABLE = DATA_DIR / "train_long.csv.gz"
VEHICLE_FEAT = DATA_DIR / "vehicle_features.csv"
ADJUSTMENTS = DATA_DIR / "test_level_adjustments.csv"
# Official outcome per vehicle (star, module score rates, test year), keyed by
# variant_id. De-identified: no model name, manufacturer or source URL.
OFFICIAL = DATA_DIR / "official_outcomes.csv"
QC_REPORT = OUT / "qc_report.json"

# ── Tier B: raw source, optional ─────────────────────────────────────────────
_raw = os.environ.get("CNCAP_RAW_DIR")
DATA = Path(_raw).expanduser().resolve() if _raw else None
RAW_AVAILABLE = DATA is not None and DATA.exists()


def require_raw(script: str) -> None:
    """Fail with an actionable message instead of a confusing FileNotFoundError."""
    if not RAW_AVAILABLE:
        raise SystemExit(
            f"\n{script} rebuilds the dataset from the raw assessment records, "
            f"which are not redistributed in this repository.\n"
            f"Set CNCAP_RAW_DIR to a local copy to enable it, or use the "
            f"shipped derivative in data/ (Tier A) for every other script.\n"
            f"See README, section 'Data provenance'.\n")


if RAW_AVAILABLE:
    LIST_CSV = DATA / "01_ListPageInformation" / "cncap_list_data.csv"
    DETAIL_CSV = (DATA / "02_DetailPageTestData" / "01_CNCAP_OverallInfo_set"
                  / "cncap_detail_overall_info_spider_data.csv")
    _CFG = DATA / "02_DetailPageTestData" / "02_CNCAP_VehicleInfo_SafetyFeatures_set"
    CFG_EARLY_DIR = _CFG / "CNCAP_VehicleInfoSafetyFeatures_2006-2018"
    CFG_LATE_ROOT = _CFG / "CNCAP_VehicleInfoSafetyFeatures_2021-2024"
    CFG_LATE_DIR = CFG_LATE_ROOT / "CNCAP_SafetyFeaturesData_2021-2024"
    CFG_LATE_BASIC = CFG_LATE_ROOT / "CNCAP_VehicleInfoSafetyFeatures_2021-2024.csv"
    MOD_DIR = DATA / "02_DetailPageTestData" / "03_CNCAP_OccProt_VRUProt_ActSafety_Set"
    MODULE_GLOBS = [
        MOD_DIR / "All_Modules_2006-2015" / "*.json",
        MOD_DIR / "All_Modules_2018" / "*.json",
        MOD_DIR / "All_Modules_2021" / "20251005版（数据已质检）" / "*.json",
        MOD_DIR / "All_Modules_2024" / "*.json",
    ]
else:
    LIST_CSV = DETAIL_CSV = CFG_EARLY_DIR = CFG_LATE_ROOT = None
    CFG_LATE_DIR = CFG_LATE_BASIC = MOD_DIR = None
    MODULE_GLOBS = []

# ── protocol constants ───────────────────────────────────────────────────────
# Scenarios modelled at test-item level: their sub-item maximum-score fields are
# unreliable because the source pages carry no maximum column for them.
TEST_LEVEL_SCENARIOS = {"child_static"}

# Single-row evaluation permitted by the protocol (pickups are assessed as
# single-row vehicles).
SINGLE_ROW = {("V6", "34")}

VERSION_MAP = {"2006": "V1", "2009": "V2", "2012": "V3", "2015": "V4",
               "2018": "V5", "2021": "V6", "2024年版": "V7"}
VORD = ["V1", "V2", "V3", "V4", "V5", "V6", "V7"]

# Records excluded in full. V7/12 has a collapsed page parse: module-level
# summary blocks appear inside the test-item list, a pedestrian-protection item
# is broadcast across all six test blocks, and the occupant-protection score
# rate appears twice with conflicting values. See the paper, Section IV-C.
BAD_RECORDS = {("V7", "12")}
