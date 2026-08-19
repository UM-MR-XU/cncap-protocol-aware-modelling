# -*- coding: utf-8 -*-
"""
确定性设计准则挖掘（T2 专题 · 由使用者在 M1 验收 A7 提出）

问题
----
并非所有细项都值得建模。有些细项在几乎所有车上都是满分（如 ecall 26/26），
标签没有变异，模型学不到东西——但这恰恰意味着它是一条**确定性的设计约束**：
想拿五星，这项就得做到。

反过来，鞭打试验只有 3.1% 的车拿满分，统计上给不出结论，那才是模型该上场的地方。

本模块按分数分布把细项与配置分成三类，分别给出可操作的产出。

三类判据
--------
设 p5 = 五星及以上车辆中该项「达成」的比例，pb = 四星及以下车辆中的比例，
lift = p5 − pb 为判别力。

    A 类  强准则     p5 ≥ 0.90 且 lift ≥ 0.20
          → 五星车几乎都做到、非五星车常做不到。**这是五星的区分点**
    B 类  行业基线   p5 ≥ 0.90 且 lift < 0.20
          → 所有车都做到了，不是区分点，但可作为入门门槛
    C 类  需建模     p5 < 0.90
          → 即使五星车也常在此失分，统计给不出结论，交给预测模型

⚠ 三条必须写进论文的限定
------------------------
1. **这是描述性的必要条件，不是因果充分条件。** 星级本身由各项得分加总而来，
   「五星车某项得分高」存在循环性。准则的正确读法是「五星车都做到了 X」，
   而**不是**「做到 X 就能拿五星」。
2. **必须按规程版本分层。** 新版本的车既配置更好、星级也更高，不分层会把
   时代进步误读为因果关系。
3. **样本量门槛。** 每组两侧各至少 MIN_N 台，否则比例不可靠，一律不出准则。

用法：  python src/design_rules.py
"""
import csv, gzip, json, sys, collections
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paths as P

MIN_N = 8            # 每侧最少样本量
P_HIGH = 0.90        # 「几乎都做到」的门槛
LIFT_STRONG = 0.20   # 判别力门槛
FULL_EPS = 1e-6      # 满分判定容差


# ══════════════════════ 数据 ══════════════════════
def load():
    with gzip.open(P.LONG_TABLE, "rt", encoding="utf-8") as fh:
        lg = pd.read_csv(fh, dtype={"variant_id": str})
    lg = lg[lg["role"] == "target"].copy()
    lg["y"] = pd.to_numeric(lg["y"], errors="coerce")
    lg = lg[lg["y"].notna()]

    # Public release: read the shipped de-identified outcome table.
    m = pd.read_csv(P.OFFICIAL, encoding="utf-8-sig", dtype=str)
    m["star"] = pd.to_numeric(m["star_rating"], errors="coerce")
    stars = dict(zip(m["variant_id"], m["star"]))

    feat = pd.read_csv(P.VEHICLE_FEAT, encoding="utf-8-sig", dtype={"variant_id": str})
    return lg, stars, feat


def is_5star(s):
    return None if pd.isna(s) else s >= 5


# ══════════════════════ 细项准则 ══════════════════════
def item_rules(lg, stars):
    lg = lg.copy()
    lg["star"] = lg["variant_id"].map(stars)
    lg = lg[lg["star"].notna()]
    lg["top"] = lg["star"] >= 5
    lg["full"] = lg["y"] >= 1 - FULL_EPS

    rows = []
    for (v, item), g in lg.groupby(["version", "item_id"]):
        hi, lo = g[g["top"]], g[~g["top"]]
        if len(hi) < MIN_N or len(lo) < MIN_N:
            continue
        p5, pb = hi["full"].mean(), lo["full"].mean()
        # 五星车在该项上的得分下限（5% 分位），比「是否满分」更细的准则形式
        q05 = float(np.percentile(hi["y"], 5))
        rows.append({
            "version": v, "item_id": item,
            "scenario": g["scenario"].iloc[0], "region": g["region"].iloc[0],
            "n_5star": len(hi), "n_below": len(lo),
            "p5_full": round(p5, 4), "pb_full": round(pb, 4),
            "lift": round(p5 - pb, 4),
            "star5_y_p05": round(q05, 4),
            "star5_y_mean": round(float(hi["y"].mean()), 4),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["class"] = np.where(d.p5_full >= P_HIGH,
                          np.where(d.lift >= LIFT_STRONG, "A 强准则", "B 行业基线"),
                          "C 需建模")
    return d.sort_values(["class", "lift"], ascending=[True, False])


# ══════════════════════ 配置准则 ══════════════════════
def config_rules(feat, stars, groups_path=None):
    fg = json.loads(P.FEATURE_GROUPS.read_text(encoding="utf-8"))
    cols = []
    for name, g in fg.items():
        if name.startswith("_") or not isinstance(g, dict) or not g.get("modeling"):
            continue
        cols += g.get("binary", []) + g.get("ordinal", [])
    cols = [c for c in cols if c in feat.columns]

    f = feat.copy()
    f["star"] = f["variant_id"].map(stars)
    f = f[f["star"].notna()]
    f["top"] = f["star"] >= 5

    rows = []
    for v, g in f.groupby("version"):
        hi, lo = g[g["top"]], g[~g["top"]]
        if len(hi) < MIN_N or len(lo) < MIN_N:
            continue
        for c in cols:
            a = pd.to_numeric(hi[c], errors="coerce").dropna()
            b = pd.to_numeric(lo[c], errors="coerce").dropna()
            if len(a) < MIN_N or len(b) < MIN_N:
                continue
            # 二值列看装配率；序数列（安全带等级）看达到最高档的比例
            top_val = max(a.max(), b.max())
            if top_val <= 0:
                continue
            p5, pb = (a >= top_val).mean(), (b >= top_val).mean()
            rows.append({
                "version": v, "feature": c, "criterion": f"≥{int(top_val)}",
                "n_5star": len(a), "n_below": len(b),
                "p5": round(float(p5), 4), "pb": round(float(pb), 4),
                "lift": round(float(p5 - pb), 4),
            })
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["class"] = np.where(d.p5 >= P_HIGH,
                          np.where(d.lift >= LIFT_STRONG, "A 强准则", "B 行业基线"),
                          "C 需建模")
    return d.sort_values(["class", "lift"], ascending=[True, False])


# ══════════════════════ 报告 ══════════════════════
def write_report(it, cf, path):
    L = []
    w = L.append
    w("# 五星车的确定性设计准则（自动生成）\n")
    w(f"> 生成：`python src/design_rules.py` ｜ 样本量门槛每侧 ≥{MIN_N} 台 ｜ "
      f"强准则判据 p5≥{P_HIGH} 且 lift≥{LIFT_STRONG}\n")
    w("> **读法：这是「五星车都做到了 X」，不是「做到 X 就能拿五星」。**"
      "星级由各项得分加总而来，二者存在循环性，故只能作为必要条件的描述。\n")

    w("\n## 一、细项准则\n")
    if it.empty:
        w("样本量不足，无可出准则。\n")
    else:
        cnt = it["class"].value_counts()
        w("| 类别 | 数量 | 含义 |")
        w("|---|---|---|")
        w(f"| A 强准则 | {cnt.get('A 强准则',0)} | 五星车几乎都满分，非五星车常失分——**这是区分点** |")
        w(f"| B 行业基线 | {cnt.get('B 行业基线',0)} | 所有车都满分，不构成区分，但是入门门槛 |")
        w(f"| C 需建模 | {cnt.get('C 需建模',0)} | 五星车也常失分，统计给不出结论，交给模型 |")

        for cls in ["A 强准则", "B 行业基线", "C 需建模"]:
            sub = it[it["class"] == cls]
            if sub.empty:
                continue
            w(f"\n### {cls}（{len(sub)} 条）\n")
            if cls == "C 需建模":
                sub = sub.sort_values("p5_full").head(20)
                w("按五星车满分率升序，取最难的 20 条。**这些是模型的主战场。**\n")
            w("| 版本 | 细项 | 五星满分率 | 非五星满分率 | 判别力 | 五星车得分下限(5%分位) | n(五星/其他) |")
            w("|---|---|---|---|---|---|---|")
            for _, r in sub.iterrows():
                w(f"| {r.version} | `{r.item_id}` | {r.p5_full:.0%} | {r.pb_full:.0%} | "
                  f"**{r.lift:+.0%}** | {r.star5_y_p05:.3f} | {r.n_5star}/{r.n_below} |")

    w("\n## 二、配置准则\n")
    if cf.empty:
        w("样本量不足，无可出准则。\n")
    else:
        for cls in ["A 强准则", "B 行业基线", "C 需建模"]:
            sub = cf[cf["class"] == cls]
            if sub.empty:
                continue
            w(f"\n### {cls}（{len(sub)} 条）\n")
            w("| 版本 | 配置 | 判据 | 五星装配率 | 非五星装配率 | 判别力 | n(五星/其他) |")
            w("|---|---|---|---|---|---|---|")
            for _, r in sub.head(25).iterrows():
                w(f"| {r.version} | `{r.feature}` | {r.criterion} | {r.p5:.0%} | {r.pb:.0%} | "
                  f"**{r.lift:+.0%}** | {r.n_5star}/{r.n_below} |")

    # ── 瓶颈部位：C 类按（工况, 部位）聚合 ──
    if not it.empty:
        c = it[it["class"] == "C 需建模"]
        if not c.empty:
            w("\n## 三、瓶颈部位：C 类细项按工况与部位聚合\n")
            w("把「五星车也拿不到满分」的细项按工况与部位归并，看规律是否集中。\n")
            agg = (c.groupby(["scenario", "region"])
                    .agg(出现版本数=("version", "nunique"),
                         五星满分率均值=("p5_full", "mean"),
                         五星得分均值=("star5_y_mean", "mean"))
                    .sort_values("五星满分率均值").reset_index())
            w("| 工况 | 部位 | 出现版本数 | 五星车满分率 | 五星车平均得分 |")
            w("|---|---|---|---|---|")
            for _, r in agg.head(12).iterrows():
                w(f"| {r.scenario} | **{r.region}** | {r.出现版本数} | "
                  f"{r.五星满分率均值:.0%} | {r.五星得分均值:.2f} |")

    # ── 区分点的版本迁移 ──
    if not cf.empty:
        a = cf[cf["class"] == "A 强准则"]
        if not a.empty:
            w("\n## 四、区分点随版本迁移\n")
            w("同一项配置在不同版本的判别力：装配率饱和后即失去区分作用，新的区分点会接替出现。\n")
            piv = cf.pivot_table(index="feature", columns="version", values="lift")
            keep = piv.index[(piv.max(axis=1) >= LIFT_STRONG)]
            piv = piv.loc[keep]
            vs = [v for v in P.VORD if v in piv.columns]
            w("| 配置 | " + " | ".join(vs) + " |")
            w("|---" * (len(vs) + 1) + "|")
            for feat_name, row in piv.iterrows():
                cells = []
                for v in vs:
                    x = row.get(v)
                    cells.append("—" if pd.isna(x) else
                                 (f"**{x:+.0%}**" if x >= LIFT_STRONG else f"{x:+.0%}"))
                w(f"| `{feat_name}` | " + " | ".join(cells) + " |")
            w("\n加粗为达到强准则门槛的格子。判别力衰减即意味着该配置已成为行业标配。\n")

    w("\n## 五、如何使用\n")
    w("- **A 类**：写入设计检查表，作为五星目标下的硬性要求。")
    w("- **B 类**：不作为差异化设计的着力点，但缺失会直接掉星，属于必保项。")
    w("- **C 类**：这些细项才需要预测模型。若模型在 C 类上不优于平凡基线，")
    w("  说明当前可观测输入不足以支撑该项预测——这本身是可报告的结论。\n")

    w("\n## 六、一处必须声明的方法学差异\n")
    w("**配置准则与细项准则的可信度不同：**\n")
    w("| | 循环性 | 说明 |")
    w("|---|---|---|")
    w("| 配置准则 | **无** | 气囊、安全带等配置不直接进入总分公式（加分项除外），")
    w("「五星车装配率高」与「星级由得分决定」之间没有恒等关系，判别力是干净的 |")
    w("| 细项准则 | **有** | 星级由各细项得分加总而来，「五星车某项满分率高」部分是同义反复。")
    w("判别力仍有意义（它说明该项**相对其他项**更能区分），但不可解读为因果 |")
    w("\n改进方向：在总分相近的车辆之间做配对比较，以剥离总分效应。此为后续工作，")
    w("当前版本未做，相关结论须按上述限定表述。\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def main():
    lg, stars, feat = load()
    it = item_rules(lg, stars)
    cf = config_rules(feat, stars)
    it.to_csv(P.OUT / "design_rules_items.csv", index=False, encoding="utf-8-sig")
    cf.to_csv(P.OUT / "design_rules_configs.csv", index=False, encoding="utf-8-sig")
    write_report(it, cf, P.OUT / "design_rules_report.md")

    print(f"细项准则 {len(it)} 条 ｜ 配置准则 {len(cf)} 条")
    if not it.empty:
        print("\n细项分类：", dict(it["class"].value_counts()))
        print("\n=== A 类强准则（细项）前 12 ===")
        a = it[it["class"] == "A 强准则"].head(12)
        for _, r in a.iterrows():
            print(f"  {r.version} {r.item_id:38s} 五星{r.p5_full:5.0%} 其他{r.pb_full:5.0%} "
                  f"判别力{r.lift:+5.0%}  n={r.n_5star}/{r.n_below}")
    if not cf.empty:
        print("\n配置分类：", dict(cf["class"].value_counts()))
        print("\n=== A 类强准则（配置）前 12 ===")
        for _, r in cf[cf["class"] == "A 强准则"].head(12).iterrows():
            print(f"  {r.version} {r.feature:28s}{r.criterion:4s} 五星{r.p5:5.0%} 其他{r.pb:5.0%} "
                  f"判别力{r.lift:+5.0%}  n={r.n_5star}/{r.n_below}")
    print(f"\n→ {P.OUT / 'design_rules_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
