# -*- coding: utf-8 -*-
"""
E3 单调性约束消融 + C 类瓶颈细项分解（W5）

两个问题：
  E3   施加单调性约束（D-056）是否在前向迁移上带来增益？重点看 F3–F5。
  C类  模型在「统计给不出结论」的瓶颈细项上表现如何？这是预测路线成败所在（F-41/F-47）。

判定标准（事前设定，见方法章 §3.5.3）：
  · 增益为正 → 采纳并报告幅度
  · 为负或不显著 → 如实报告，并据已发现的两处反例说明适用边界

用法：  python src/ablation_e3.py
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paths as P
from dataset import build_design_matrix
from cv_protocol import forward_transfer_folds, shared_item_subset, metrics, skill_score
from baselines import B2VehicleGBDT, monotone_vector, _meta_slice


def load_c_items():
    """design_rules 产出的 C 类（需建模）细项；取任何门槛下均属 C 类的核心集。"""
    p = P.OUT / "design_rules_items.csv"
    if not p.exists():
        # Fail loudly. Returning empty sets here would silently drop the
        # bottleneck stratum and produce numbers that differ from the paper
        # without any warning -- a worse failure than stopping.
        raise SystemExit(
            "\nablation_e3.py needs outputs/design_rules_items.csv.\n"
            "Run  python src/design_rules.py  first (or use run_local.py, "
            "which orders the steps correctly).\n")
    d = pd.read_csv(p, encoding="utf-8-sig")
    core = set(d[d.p5_full < 0.80].item_id)      # 核心瓶颈（门槛稳健）
    wide = set(d[d["class"] == "C 需建模"].item_id)
    return core, wide


def run(seed=42):
    D = build_design_matrix()
    X, y, iid = D["X"], D["y"], D["item_id"]
    cols = X.columns.tolist()
    core_c, wide_c = load_c_items()
    print(f"C 类核心瓶颈细项 {len(core_c)} 个 ｜ C 类全体 {len(wide_c)} 个\n")

    rows = []
    for use_mono in (False, True):
        # B2 会在设计矩阵末尾追加 b1_item_prior 一列，故单调向量须同步扩展
        mono = monotone_vector(cols + ["b1_item_prior"], enable=use_mono)
        tag = "有约束" if use_mono else "无约束"
        b2 = B2VehicleGBDT(seed)
        b2.fit_predict, b2.backend = __import__("baselines")._get_gbdt(seed, mono)
        for nm, tr, te, tv, tev in forward_transfer_folds(D["version"]):
            t0 = time.time()
            mtr, mte = _meta_slice(D, tr), _meta_slice(D, te)
            pred, _ = b2.fit_predict_fold(X, y, tr, te, mtr, mte, D["categorical_cols"])
            sub, _ = shared_item_subset(iid, tr, te)
            yt, it_ = y[te], iid[te]
            for scope, idx in (("全部 item", np.arange(len(yt))),
                               ("共有子集", sub),
                               ("C 类核心", np.where(np.isin(it_, list(core_c)))[0])):
                if idx.size < 5:
                    continue
                m = metrics(yt[idx], pred[idx])
                rows.append({"约束": tag, "fold": nm, "test": tev, "scope": scope,
                             "n": m["n"], "MAE": round(m["MAE"], 4),
                             "RMSE": round(m["RMSE"], 4),
                             "skill_MAE": round(skill_score(m, "MAE"), 4),
                             "skill_RMSE": round(skill_score(m, "RMSE"), 4)})
            print(f"  [{tag}] {nm} → {tev}  {time.time()-t0:.0f}s")
    d = pd.DataFrame(rows)
    d.to_csv(P.OUT / "ablation_e3.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("E3　单调性约束的增益（skill_MAE，正值为优）")
    print("=" * 78)
    for scope in ["全部 item", "共有子集", "C 类核心"]:
        s = d[d.scope == scope]
        if s.empty:
            continue
        piv = s.pivot_table(index="fold", columns="约束", values="skill_MAE")
        if "有约束" not in piv or "无约束" not in piv:
            continue
        piv["增益"] = piv["有约束"] - piv["无约束"]
        print(f"\n--- {scope} ---")
        print(piv.round(4).to_string())
        late = piv.loc[[f for f in ["F3", "F4", "F5"] if f in piv.index]]
        print(f"  F3–F5 平均增益：{late['增益'].mean():+.4f}"
              f"　→ {'✔ 采纳' if late['增益'].mean() > 0.01 else '✘ 无显著增益'}")

    print("\n" + "=" * 78)
    print("C 类瓶颈细项上的表现（模型的主战场）")
    print("=" * 78)
    c = d[(d.scope == "C 类核心") & (d.约束 == "无约束")]
    if not c.empty:
        print(c[["fold", "test", "n", "MAE", "RMSE", "skill_MAE", "skill_RMSE"]].to_string(index=False))
    print(f"\n→ {P.OUT / 'ablation_e3.csv'}")
    return d


if __name__ == "__main__":
    run()
