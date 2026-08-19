# -*- coding: utf-8 -*-
"""
oracle / zero 双模式评估 + E4 可观测性排序（方法章 §3.2.4、§3.5.4 承诺项）

要回答什么
----------
官方试验项得分 ≠ 各细项得分之和，差额为**试验项级调整**（罚分 162 组、加分 24 组、
条件置零 2 组）。该调整量只见于页面注释文本，训练时可由「官方得分 − 细项和」反解，
但**预测一台未测车时无从得知**。

因此同一个模型有两种评估口径：

    oracle   使用观测到的调整量 → 性能上界（假装我们知道会不会被罚分）
    zero     假定调整量为零     → 预测阶段实际可达

**两者之差即「不可观测成分」的代价，本身作为结果报告。**

评估层级的选择（一处实事求是的退让）
------------------------------------
理想是评估到星级，但 V5–V7 的综合得分率需各模块权重，而规程记录不全
（`rule_config.module_max` 仅有零星条目）。故：

  · **主评估在「乘员保护模块得分率」层面**——规程明确定义的量，七版皆可算，
    且 V5 起的分模块门槛正是对它设定的
  · **补充评估到星级，限 V1–V4**——该四版无 VRU 与主动安全模块，
    判定量即绝对总分，总分完全由乘员保护细项决定，可做真正的端到端

三条线的含义
------------
    上界      真实细项 + 观测调整 → 应精确等于官方（验证链路无误）
    oracle    预测细项 + 观测调整 → 细项预测误差的代价
    zero      预测细项 + 零调整   → 再叠加不可观测调整的代价

用法：  python src/oracle_zero.py
"""
import gzip, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paths as P
from dataset import build_design_matrix
from cv_protocol import forward_transfer_folds, metrics, skill_score
from baselines import B2VehicleGBDT, _meta_slice
import rule_layer as RL


def load_adjustments():
    """(variant_id, scenario) → 调整量；以及官方试验项得分与分母。"""
    # Public release: this table is shipped in data/ (P.ADJUSTMENTS). In the
    # working tree it was a product of build_long_table.py and therefore lived
    # in outputs/; fall back to that location so both layouts work.
    p = P.ADJUSTMENTS if P.ADJUSTMENTS.exists() else P.OUT / "test_level_adjustments.csv"
    if not p.exists():
        raise SystemExit(
            "\noracle_zero.py needs test_level_adjustments.csv.\n"
            f"Looked in {P.ADJUSTMENTS} and {P.OUT / 'test_level_adjustments.csv'}.\n")
    d = pd.read_csv(p, encoding="utf-8-sig", dtype={"variant_id": str})
    adj = {(r.variant_id, r.scenario): float(r.adjustment) for r in d.itertuples()}
    off = {(r.variant_id, r.scenario): (float(r.official_test_score), float(r.denominator))
           for r in d.itertuples()}
    return adj, off


def module_rate(item_scores, denoms, adj_map, vid, use_adj):
    """由细项得分算乘员保护模块得分率。

    item_scores: {(scenario): 该工况细项得分之和}
    denoms:      {(scenario): 官方分母}
    """
    num = den = 0.0
    for scn, s in item_scores.items():
        if scn not in denoms:
            continue
        a = adj_map.get((vid, scn), 0.0) if use_adj else 0.0
        num += max(0.0, s + a)          # 得分不为负
        den += denoms[scn]
    return (num / den) if den > 0 else np.nan


def run(seed=42):
    D = build_design_matrix()
    X, y, iid = D["X"], D["y"], D["item_id"]
    adj_map, off_map = load_adjustments()

    # 长表中每行的满分，用于把归一化预测还原为分数
    with gzip.open(P.LONG_TABLE, "rt", encoding="utf-8") as fh:
        lg = pd.read_csv(fh, dtype={"variant_id": str})
    lg = lg[lg["role"] == "target"].copy()
    lg["max"] = pd.to_numeric(lg["max"], errors="coerce")
    lg["y"] = pd.to_numeric(lg["y"], errors="coerce")
    lg = lg[lg["y"].notna()].reset_index(drop=True)
    assert len(lg) == len(y), "长表与设计矩阵行数不一致"
    maxes = lg["max"].to_numpy()
    vids = lg["variant_id"].to_numpy()
    scns = lg["scenario"].to_numpy()

    rows, star_rows = [], []
    b2 = B2VehicleGBDT(seed)
    print(f"后端：{b2.backend}\n")

    for nm, tr, te, tv, tev in forward_transfer_folds(D["version"]):
        t0 = time.time()
        mtr, mte = _meta_slice(D, tr), _meta_slice(D, te)
        pred, _ = b2.fit_predict_fold(X, y, tr, te, mtr, mte, D["categorical_cols"])

        # 按 (车, 工况) 聚合：真实与预测的细项得分之和
        sub = pd.DataFrame({
            "vid": vids[te], "scn": scns[te],
            "true_s": y[te] * maxes[te], "pred_s": pred * maxes[te],
            "den": maxes[te],
        })
        g = sub.groupby(["vid", "scn"]).agg(true_s=("true_s", "sum"),
                                            pred_s=("pred_s", "sum"),
                                            den=("den", "sum")).reset_index()

        recs = []
        for vid, gv in g.groupby("vid"):
            d_true = dict(zip(gv.scn, gv.true_s))
            d_pred = dict(zip(gv.scn, gv.pred_s))
            denoms = dict(zip(gv.scn, gv.den))
            # 官方模块得分率：用官方试验项得分（含调整）与官方分母
            num_off = sum(off_map.get((vid, s), (np.nan, 0))[0] for s in denoms)
            den_off = sum(off_map.get((vid, s), (0, np.nan))[1] for s in denoms)
            if not np.isfinite(num_off) or not den_off:
                continue
            recs.append({
                "vid": vid,
                "official": num_off / den_off,
                "upper": module_rate(d_true, denoms, adj_map, vid, True),
                "oracle": module_rate(d_pred, denoms, adj_map, vid, True),
                "zero": module_rate(d_pred, denoms, adj_map, vid, False),
            })
        r = pd.DataFrame(recs).dropna()
        if r.empty:
            continue
        base = np.abs(r.official - r.official.mean()).mean()
        for mode in ("upper", "oracle", "zero"):
            e = np.abs(r[mode] - r.official)
            rows.append({"fold": nm, "test": tev, "n": len(r), "mode": mode,
                         "MAE": round(float(e.mean()), 4),
                         "RMSE": round(float(np.sqrt(((r[mode]-r.official)**2).mean())), 4),
                         "max_err": round(float(e.max()), 4),
                         "skill_vs_mean": round(1 - float(e.mean()) / base, 4) if base > 0 else np.nan})

        # ── 补充：V1–V4 端到端星级（该四版总分即乘员保护，无需模块权重）──
        # ⚠ 总分 = 各碰撞工况得分 + 加分项。加分项由安全配置确定性给出（D-012，
        #   role=rule_unit_test，不在回归目标内），故不参与预测，须按官方值加回；
        #   否则 upper 口径也会系统性低估总分（初版即因此只有 87–90%）。
        if tev in ("V3", "V4"):
            # Public release: official outcomes come from the shipped
            # de-identified table, keyed directly by variant_id.
            lst = {x["variant_id"]: x for x in RL.read_csv(P.OFFICIAL)}
            den_by_vid = g.groupby("vid").den.sum().to_dict()
            acc = {}
            for mode in ("upper", "oracle", "zero"):
                ok = n = 0
                for _, rr in r.iterrows():
                    m0 = lst.get(rr.vid)
                    if not m0:
                        continue
                    official_star = RL.f(m0.get("star_rating"))
                    official_total = RL.f(m0.get("overall_score"))
                    den_v = den_by_vid.get(rr.vid)
                    if official_star is None or official_total is None or not den_v:
                        continue
                    # 加分项 = 官方总分 − 官方各工况得分之和（确定性量，按官方值加回）
                    bonus = official_total - rr.official * den_v
                    pred_total = rr[mode] * den_v + bonus
                    s, _ = RL.decide_star(tev, pred_total)
                    ok += int(s == official_star); n += 1
                acc[mode] = (ok / n, n) if n else (np.nan, 0)
            star_rows.append({"fold": nm, "test": tev, "n": acc["upper"][1],
                              **{f"{k}_acc": round(v[0], 4) for k, v in acc.items()}})
        print(f"  {nm} → {tev}  {time.time()-t0:.0f}s  n={len(r)}")

    d = pd.DataFrame(rows)
    d.to_csv(P.OUT / "oracle_zero.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("乘员保护模块得分率的预测误差（相对官方值）")
    print("=" * 80)
    piv = d.pivot_table(index=["fold", "test"], columns="mode", values="MAE")
    piv = piv[["upper", "oracle", "zero"]]
    piv["细项预测代价"] = (piv["oracle"] - piv["upper"]).round(4)
    piv["不可观测调整代价"] = (piv["zero"] - piv["oracle"]).round(4)
    print(piv.round(4).to_string())
    print(f"\n  链路自检：upper 应接近 0（真实细项 + 观测调整 = 官方）"
          f"　实测最大 {piv['upper'].max():.4f}")
    print(f"  不可观测调整的平均代价：{piv['不可观测调整代价'].mean():+.4f} "
          f"（占 zero 模式总误差的 {100*piv['不可观测调整代价'].mean()/piv['zero'].mean():.0f}%）")

    if star_rows:
        sd = pd.DataFrame(star_rows)
        sd.to_csv(P.OUT / "oracle_zero_star.csv", index=False, encoding="utf-8-sig")
        print("\n" + "=" * 80)
        print("端到端星级准确率（限 V1–V4：总分即乘员保护，无需模块权重）")
        print("=" * 80)
        print(sd.to_string(index=False))

    # ── E4：可观测性三分层 ──
    print("\n" + "=" * 80)
    print("E4　可观测性三分层的预测能力（分组 5 折，同分布）")
    print("=" * 80)
    from cv_protocol import group_kfold_indices
    obs = D["observability"]
    pred_all = np.zeros(len(y))
    for tr, te in group_kfold_indices(D["groups"], 5, seed):
        mtr, mte = _meta_slice(D, tr), _meta_slice(D, te)
        pred_all[te], _ = b2.fit_predict_fold(X, y, tr, te, mtr, mte, D["categorical_cols"])
    e4 = []
    for cls in ["config_gated", "structure_driven", "component_driven"]:
        m = obs == cls
        if m.sum() < 5:
            continue
        mm = metrics(y[m], pred_all[m])
        e4.append({"可观测性": cls, "n": int(m.sum()),
                   "MAE": round(mm["MAE"], 4), "skill_MAE": round(skill_score(mm, "MAE"), 4),
                   "满分率": round(mm["full_score_rate"], 3)})
    e4 = pd.DataFrame(e4)
    e4.to_csv(P.OUT / "e4_observability.csv", index=False, encoding="utf-8-sig")
    print(e4.to_string(index=False))
    print("\n  事前假设的排序：config_gated > structure_driven > component_driven")
    print(f"  实测 skill 排序：{' > '.join(e4.sort_values('skill_MAE', ascending=False)['可观测性'])}")
    print(f"\n→ {P.OUT / 'oracle_zero.csv'}　{P.OUT / 'e4_observability.csv'}")
    return d, e4


if __name__ == "__main__":
    run()
