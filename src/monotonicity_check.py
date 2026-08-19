# -*- coding: utf-8 -*-
"""
单调性假设的检验（T6 专题 · 由使用者提出）

待检假设
--------
「C-NCAP 版本更新使安全要求越来越严苛，因此某版 5 星的必要条件，
  是后续版本 5 星必要条件的子集——要求只会继承或提高，不会降低。」

若成立，它是跨版本迁移最有价值的结构性先验：可作为单调性约束纳入模型，
并为小样本版本（V7 仅 14 台）提供外推依据。

但这是一个经验命题，不是公理。本模块把它拆成五个可证伪的子命题分别检验，
**任何一处反例都必须显式记录，而不是忽略。**

    M1  σ 严酷度单调不减        碰撞速度 / Δv 是否只增不减
    M2  归一化星级门槛单调不减  五星线 ÷ 总分，跨版本是否只升不降
    M3  五星车配置装配率单调不减 同一配置，五星车中的装配率是否只升不降
    M4  五星车细项得分下限单调不减 同一细项，五星车得分的 5% 分位是否只升不降
    M5  有序特征的分布单调上移   如后排安全带等级，五星车中的分布是否右移

三条必须区分的层面（混淆会得出错误结论）
----------------------------------------
  · **规程层面**：条款是否收紧（M1、M2 —— 查原文可判定）
  · **数据层面**：实际车辆是否改善（M3、M4、M5 —— 统计可判定）
  · **替换 vs 继承**：ODB40 → MPDB50 是工况**替换**而非提高，
    此处单调性不适用，须先按「是否存在跨版本对应」筛选

用法：  python src/monotonicity_check.py
"""
import gzip, json, sys, collections
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paths as P

MIN_N = 8


def load():
    with gzip.open(P.LONG_TABLE, "rt", encoding="utf-8") as fh:
        lg = pd.read_csv(fh, dtype={"variant_id": str})
    lg = lg[lg["role"] == "target"].copy()
    lg["y"] = pd.to_numeric(lg["y"], errors="coerce")
    lg = lg[lg["y"].notna()]
    # Public release: read the shipped de-identified outcome table.
    m = pd.read_csv(P.OFFICIAL, encoding="utf-8-sig", dtype=str)
    stars = dict(zip(m["variant_id"], pd.to_numeric(m["star_rating"], errors="coerce")))
    feat = pd.read_csv(P.VEHICLE_FEAT, encoding="utf-8-sig", dtype={"variant_id": str})
    onto = json.loads(P.ONTO.read_text(encoding="utf-8"))
    cfg = json.loads(P.RULE_CONFIG.read_text(encoding="utf-8"))
    return lg, stars, feat, onto, cfg


def vseq(present):
    return [v for v in P.VORD if v in present]


def check_monotone(pairs):
    """pairs: [(version, value)]，按版本序检查是否单调不减。返回 (结论, 违例列表)"""
    pairs = [(v, x) for v, x in pairs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    bad = []
    for i in range(1, len(pairs)):
        if pairs[i][1] < pairs[i - 1][1] - 1e-9:
            bad.append(f"{pairs[i-1][0]}({pairs[i-1][1]:.4g}) → {pairs[i][0]}({pairs[i][1]:.4g})")
    return (len(bad) == 0), bad


# ══════════════ M1  σ 严酷度 ══════════════
def m1_sigma(onto):
    st = onto["sigma_table"]
    rows = []
    for scn, tbl in st.items():
        for dim in ("speed_kmh", "delta_v_kmh"):
            if dim not in tbl:
                continue
            pairs = [(v, tbl[dim].get(v)) for v in P.VORD if v in tbl[dim]]
            pairs = [(v, x) for v, x in pairs if isinstance(x, (int, float))]
            if len(pairs) < 2:
                continue
            ok, bad = check_monotone(pairs)
            rows.append({"工况": scn, "维度": dim,
                         "取值序列": " → ".join(f"{v}:{x:g}" for v, x in pairs),
                         "单调不减": "✔" if ok else "✘",
                         "反例": "；".join(bad) or "—"})
        # 假人变更（不赋全序，只记录是否更换）
        for dk in ("row1_dummy", "row2_dummy", "dummy"):
            if dk not in tbl:
                continue
            pairs = [(v, tbl[dk].get(v)) for v in P.VORD if v in tbl[dk]]
            ch = [f"{pairs[i-1][0]}→{pairs[i][0]}: {pairs[i-1][1]} ⇒ {pairs[i][1]}"
                  for i in range(1, len(pairs)) if pairs[i][1] != pairs[i - 1][1]]
            if ch:
                rows.append({"工况": scn, "维度": dk, "取值序列": "；".join(ch),
                             "单调不减": "—（假人不赋全序）", "反例": "见左"})
    return pd.DataFrame(rows)


# ══════════════ M2  归一化星级门槛 ══════════════
def m2_threshold(cfg, lg):
    """五星线 ÷ 该版本总分。绝对分制需知道总分，得分率制直接就是比例。"""
    # 绝对分制版本的总分：取该版本各 item 满分之和的众数（含非 target 行更全，此处用官方分母近似）
    tot = {}
    for v in ("V1", "V2", "V3", "V4"):
        sub = lg[lg["version"] == v]
        if sub.empty:
            continue
        per = sub.groupby("variant_id")["max"].apply(lambda s: pd.to_numeric(s, errors="coerce").sum())
        tot[v] = float(per.mode().iloc[0]) if len(per) else None
    rows = []
    for v in P.VORD:
        th = cfg["star_thresholds"].get(v)
        if not th:
            continue
        five = next((lo for lo, star in th if star == 5), None)
        if five is None:
            continue
        if v in tot and tot[v]:
            norm = five / tot[v]
            note = f"{five:g} / {tot[v]:g}（细项满分和）"
        elif five <= 1:
            norm, note = five, "得分率制，直接可比"
        else:
            norm, note = None, "总分未知"
        rows.append({"版本": v, "五星线": five, "归一化": None if norm is None else round(norm, 4),
                     "口径": note})
    d = pd.DataFrame(rows)
    ok, bad = check_monotone([(r["版本"], r["归一化"]) for _, r in d.iterrows()])
    return d, ok, bad


# ══════════════ M3  五星车配置装配率 ══════════════
def m3_config(feat, stars):
    fg = json.loads(P.FEATURE_GROUPS.read_text(encoding="utf-8"))
    cols = []
    for name, g in fg.items():
        if name.startswith("_") or not isinstance(g, dict) or not g.get("modeling"):
            continue
        cols += g.get("binary", []) + g.get("ordinal", [])
    cols = [c for c in cols if c in feat.columns]
    f = feat.copy()
    f["star"] = f["variant_id"].map(stars)
    f = f[(f["star"].notna()) & (f["star"] >= 5)]
    rows = []
    for c in cols:
        pairs = []
        for v in P.VORD:
            g = f[f["version"] == v]
            a = pd.to_numeric(g[c], errors="coerce").dropna()
            if len(a) < MIN_N:
                continue
            top = a.max()
            pairs.append((v, float((a >= max(top, 1)).mean()) if top >= 1 else 0.0))
        if len(pairs) < 2:
            continue
        ok, bad = check_monotone(pairs)
        rows.append({"配置": c, "五星车达成率序列": " → ".join(f"{v}:{x:.0%}" for v, x in pairs),
                     "单调不减": "✔" if ok else "✘", "反例": "；".join(bad) or "—"})
    return pd.DataFrame(rows)


# ══════════════ M4  五星车细项得分下限 ══════════════
def m4_item_floor(lg, stars, q=5):
    lg = lg.copy()
    lg["star"] = lg["variant_id"].map(stars)
    hi = lg[(lg["star"].notna()) & (lg["star"] >= 5)]
    rows = []
    for item, g in hi.groupby("item_id"):
        pairs = []
        for v in P.VORD:
            s = g[g["version"] == v]["y"]
            if len(s) < MIN_N:
                continue
            pairs.append((v, float(np.percentile(s, q))))
        if len(pairs) < 2:
            continue
        ok, bad = check_monotone(pairs)
        rows.append({"细项": item, "跨版本数": len(pairs),
                     f"五星车得分{q}%分位序列": " → ".join(f"{v}:{x:.2f}" for v, x in pairs),
                     "单调不减": "✔" if ok else "✘", "反例": "；".join(bad) or "—"})
    return pd.DataFrame(rows).sort_values(["单调不减", "跨版本数"], ascending=[True, False])


# ══════════════ M5  有序特征分布 ══════════════
def m5_ordinal(feat, stars, col="belt_grade_row2"):
    if col not in feat.columns:
        return pd.DataFrame()
    f = feat.copy()
    f["star"] = f["variant_id"].map(stars)
    f = f[(f["star"].notna()) & (f["star"] >= 5)]
    rows = []
    for v in P.VORD:
        a = pd.to_numeric(f[f["version"] == v][col], errors="coerce").dropna()
        if len(a) < MIN_N:
            continue
        rows.append({"版本": v, "n": len(a), "均值": round(float(a.mean()), 3),
                     "中位数": float(a.median()), "最低档占比": round(float((a == 0).mean()), 3),
                     "最高档占比": round(float((a == a.max()).mean()), 3)})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    ok, bad = check_monotone([(r["版本"], r["均值"]) for _, r in d.iterrows()])
    d.attrs["ok"], d.attrs["bad"] = ok, bad
    return d


def main():
    lg, stars, feat, onto, cfg = load()
    out = {}

    print("=" * 78)
    print("M1　σ 严酷度是否单调不减（规程层面）")
    print("=" * 78)
    d1 = m1_sigma(onto)
    print(d1.to_string(index=False))
    out["M1"] = d1.to_dict("records")

    print("\n" + "=" * 78)
    print("M2　归一化五星门槛是否单调不减（规程层面）")
    print("=" * 78)
    d2, ok2, bad2 = m2_threshold(cfg, lg)
    print(d2.to_string(index=False))
    print(f"\n结论：{'✔ 单调不减' if ok2 else '✘ 存在反例'}")
    for b in bad2:
        print("   反例：", b)
    out["M2"] = {"table": d2.to_dict("records"), "monotone": ok2, "violations": bad2}

    print("\n" + "=" * 78)
    print("M3　五星车配置装配率是否单调不减（数据层面）")
    print("=" * 78)
    d3 = m3_config(feat, stars)
    print(d3.to_string(index=False))
    out["M3"] = d3.to_dict("records")

    print("\n" + "=" * 78)
    print("M4　五星车细项得分下限是否单调不减（数据层面）")
    print("=" * 78)
    d4 = m4_item_floor(lg, stars)
    n_ok = int((d4["单调不减"] == "✔").sum()) if not d4.empty else 0
    print(f"可比细项 {len(d4)} 个：单调 {n_ok}，违例 {len(d4)-n_ok}")
    print("\n--- 违例（前 12）---")
    print(d4[d4["单调不减"] == "✘"].head(12).to_string(index=False))
    print("\n--- 单调（前 8）---")
    print(d4[d4["单调不减"] == "✔"].head(8).to_string(index=False))
    out["M4"] = {"n_total": len(d4), "n_monotone": n_ok,
                 "records": d4.to_dict("records")}

    print("\n" + "=" * 78)
    print("M5　后排安全带等级在五星车中的分布（使用者所举之例）")
    print("=" * 78)
    d5 = m5_ordinal(feat, stars)
    if d5.empty:
        print("样本不足")
    else:
        print(d5.to_string(index=False))
        print(f"\n均值单调不减：{'✔' if d5.attrs['ok'] else '✘ ' + '；'.join(d5.attrs['bad'])}")
    out["M5"] = d5.to_dict("records") if not d5.empty else []

    (P.OUT / "monotonicity_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for nm, d in [("m1_sigma", d1), ("m2_threshold", d2), ("m3_config", d3),
                  ("m4_item_floor", d4), ("m5_ordinal", d5)]:
        if not d.empty:
            d.to_csv(P.OUT / f"mono_{nm}.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {P.OUT / 'monotonicity_check.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
