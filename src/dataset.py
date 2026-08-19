# -*- coding: utf-8 -*-
"""
建模数据集组装（W4 交付物）

将长表（每行 = 车辆 × 细项）与车辆特征表连接，产出模型可直接消费的设计矩阵。

    X = [ 车辆特征 ⊕ item 本体三元组 ⊕ σ 测量条件签名 ]
    y = score / max ∈ [0,1]
    groups = variant_id      （分组 CV 用，防同车泄漏）
    version = V1..V7         （前向迁移切分用）

三条硬性约定
------------
1. **入模特征一律从 `config/feature_groups.json` 读取**，代码内不硬编码特征名单。
   主动安全配置、cfg_ecall、cfg_active_bonnet 均已在该文件中声明为不入模（D-047）。
2. **version 与 test_year 不作为特征**。前向迁移评估中二者与折划分共线，
   入模会让模型直接记忆版本而非学习跨版本结构。
3. **不做缺失值填充**。缺失以 NaN 传递给模型，由 GBDT 原生处理。
   填 0 会把「该版本不评价此项」与「配备了但没有」混为一谈（D-046 作废的原因）。

用法
----
    from dataset import build_design_matrix
    D = build_design_matrix()                       # 默认特征集
    D = build_design_matrix(include_component=True) # 加入部件级族系代理（仅 V3–V5 可得）
    D = build_design_matrix(roles=("target",))      # 默认即此
"""
import json, sys, gzip
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paths as P


# ══════════════════════════ 特征分组 ══════════════════════════
def load_feature_groups():
    return json.loads(P.FEATURE_GROUPS.read_text(encoding="utf-8"))


def resolve_feature_columns(include_component=False):
    """按 feature_groups.json 解析入模列，返回 (数值列, 类别列)。

    include_component=False 时排除 `component_proxy`（仅 V3–V5 可得，见 F-34）：
    默认主实验不含，以免前向迁移结果被特征可得性差异混淆。
    """
    fg = load_feature_groups()
    num, cat = [], []
    for name, g in fg.items():
        if name.startswith("_") or not isinstance(g, dict):
            continue
        if not g.get("modeling"):
            continue
        if g.get("excluded_from_default") and not include_component:
            continue
        num += g.get("numeric", []) + g.get("binary", []) + g.get("ordinal", [])
        cat += g.get("categorical", [])
    return num, cat


def excluded_columns():
    """全部声明为不入模的列——用于断言设计矩阵中确无泄漏列。"""
    fg = load_feature_groups()
    out = []
    for name, g in fg.items():
        if name.startswith("_") or not isinstance(g, dict):
            continue
        out += g.get("excluded", [])
    return out


# ══════════════════════════ 载入 ══════════════════════════
def load_long():
    with gzip.open(P.LONG_TABLE, "rt", encoding="utf-8") as fh:
        d = pd.read_csv(fh, dtype={"variant_id": str, "test_year": str})
    for c in ("score", "max", "y", "sigma_speed"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def load_vehicle_features():
    d = pd.read_csv(P.VEHICLE_FEAT, encoding="utf-8-sig", dtype={"variant_id": str})
    return d


# ══════════════════════════ 组装 ══════════════════════════
# item 本体三元组：长表中已分解好的结构特征，是「长表编码」的实现核心——
# 模型据此对未见过的 item 组合外推，而非依赖固定输出头。
ONTOLOGY_CAT = ["scenario", "dummy", "region", "observability_class"]
SIGMA_CAT = ["sigma_barrier", "sigma_dummy"]
SIGMA_NUM = ["sigma_speed"]


def build_design_matrix(roles=("target",), include_component=False,
                        drop_config_gated=False):
    """返回一个 dict，含 X（DataFrame）、y、groups、version 与列元信息。

    参数
    ----
    roles              保留哪些 role 的行。默认仅 target（回归目标）。
    include_component  是否加入部件级族系代理（belt/airbag/seat supplier）。
    drop_config_gated  是否剔除 config_gated 类细项。该类由配置确定性决定，
                       预测精度平凡（D-047），报告主指标时建议单列而非混入。
    """
    long = load_long()
    veh = load_vehicle_features()

    long = long[long["role"].isin(roles)].copy()
    if drop_config_gated:
        long = long[long["observability_class"] != "config_gated"].copy()
    long = long[long["y"].notna()].copy()

    num_cols, cat_cols = resolve_feature_columns(include_component)
    keep = ["variant_id"] + [c for c in num_cols + cat_cols if c in veh.columns]
    missing = [c for c in num_cols + cat_cols if c not in veh.columns]
    veh = veh[keep]

    d = long.merge(veh, on="variant_id", how="left", validate="many_to_one")
    assert len(d) == len(long), "连接后行数变化——vehicle_features 的 variant_id 应唯一"

    num = [c for c in num_cols if c in d.columns] + SIGMA_NUM
    cat = [c for c in cat_cols if c in d.columns] + ONTOLOGY_CAT + SIGMA_CAT
    num = [c for c in num if c in d.columns]
    cat = [c for c in cat if c in d.columns]

    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in cat:
        d[c] = d[c].astype("category")

    X = d[num + cat]
    y = d["y"].to_numpy(dtype=float)
    groups = d["variant_id"].to_numpy()
    version = d["version"].to_numpy()

    # ── 泄漏与范围自检 ──
    bad = [c for c in excluded_columns() if c in X.columns]
    assert not bad, f"设计矩阵含已声明不入模的列（泄漏或越界）：{bad}"
    for c in ("version", "test_year", "variant_id", "item_id", "score", "max", "y"):
        assert c not in X.columns, f"设计矩阵不得含 {c}"
    assert np.isfinite(y).all() and (y >= -1e-9).all() and (y <= 1 + 1e-9).all(), "标签越界"

    return {
        "X": X, "y": y, "groups": groups, "version": version,
        "item_id": d["item_id"].to_numpy(), "scenario": d["scenario"].to_numpy(),
        "observability": d["observability_class"].to_numpy(),
        "numeric_cols": num, "categorical_cols": cat,
        "missing_declared_cols": missing,
        "n_rows": len(d), "n_vehicles": d["variant_id"].nunique(),
    }


def summarize(D):
    X = D["X"]
    lines = [
        f"行数 {D['n_rows']}｜车辆 {D['n_vehicles']}｜特征 {X.shape[1]}"
        f"（数值 {len(D['numeric_cols'])}，类别 {len(D['categorical_cols'])}）",
        f"标签 y：均值 {D['y'].mean():.4f}，标准差 {D['y'].std():.4f}，"
        f"= 1.0 的比例 {np.mean(np.isclose(D['y'], 1.0)):.1%}",
    ]
    vc = pd.Series(D["version"]).value_counts().sort_index()
    lines.append("按版本行数：" + "，".join(f"{k} {v}" for k, v in vc.items()))
    miss = X.isna().mean().sort_values(ascending=False)
    hi = miss[miss > 0.3]
    if len(hi):
        lines.append("缺失率 >30% 的特征：" + "，".join(f"{k} {v:.0%}" for k, v in hi.items()))
    if D["missing_declared_cols"]:
        lines.append("⚠ 声明入模但特征表中不存在的列：" + "，".join(D["missing_declared_cols"]))
    return "\n".join(lines)


if __name__ == "__main__":
    D = build_design_matrix()
    print("=== 默认特征集 ===")
    print(summarize(D))
    print("\n=== 含部件级族系代理 ===")
    print(summarize(build_design_matrix(include_component=True)))
