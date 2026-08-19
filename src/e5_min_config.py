# -*- coding: utf-8 -*-
"""
E5　面向目标星级的最小充分配置集（N3 贡献点的载体）

问题形式
--------
    argmin  cost(x)   s.t.   g_v( f(x, ·) ) ≥ 目标

其中 cost 取被动安全配置项数（F-49 已验证：与价格秩相关仅 0.37–0.65，
控制价格后仍能区分星级，故携带价格之外的信息）。

为什么不在配置空间穷举
----------------------
11 个二值配置 + 2 个三档序数 ≈ 1.8 万种组合，穷举本身可行；但其中绝大多数
**从未在历史数据中出现**，模型在那里是纯外推，不可信。

改用**逐项反事实消融**：对每台真实存在的车，逐一问「若去掉这项配置，还达标吗」。
每次只动一个维度，样本始终落在数据流形附近。

    基线      r_i        = 模块得分率（模型预测）
    反事实    r_i^(-c)   = 把配置 c 置为 0（序数降至 0）后重新预测
    必要性    c 对车 i 必要  ⟺  r_i ≥ 门槛 且 r_i^(-c) < 门槛

⚠ 三条必须声明的限制
--------------------
1. **这是模型内的反事实**，反映模型学到的关联，**不等于真实因果效应**。
   571 台观测数据、无随机化无干预，不满足因果识别条件（D-003）。
2. 门槛取乘员保护模块得分率，而非星级——V5–V7 的综合得分率需模块权重，
   规程记录不全（同 oracle_zero.py 的处置）。V5 起规程对该模块直接设有门槛，
   四星 ≥75%、五星 ≥85%，故该口径有规程依据。
3. 配置项之间可能存在协同（如气帘与侧气囊），单项消融无法捕捉。
   本模块给出的是**单项必要性**，不是最优组合。

用法：  python src/e5_min_config.py
"""
import gzip, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paths as P
from dataset import build_design_matrix
from baselines import B2VehicleGBDT, _meta_slice

# 门槛的选取（初版取 0.75/0.85，实测失效——见下）
#   真实模块得分率中位数 0.908，0.75 门槛下仅 10.4% 的车不达标，而单项配置的
#   边际贡献最大仅约 0.056，够不着门槛，必要率恒为 0。
#   故改用：0.85（V5 起规程明载的五星门槛）与 0.90（样本最密集处，余量<0.06 的
#   达标车达 271 台，是单项消融唯一可能击穿的区间）。
THRESHOLDS = {"五星门槛": 0.85, "高余量": 0.90}
MIN_N = 8


def config_columns():
    fg = json.loads(P.FEATURE_GROUPS.read_text(encoding="utf-8"))
    g = fg.get("passive_safety", {})
    return g.get("binary", []), g.get("ordinal", [])


def module_rates(pred, vids, scns, maxes, idx):
    """把细项预测聚合为每车的乘员保护模块得分率。"""
    d = pd.DataFrame({"vid": vids[idx], "s": pred * maxes[idx], "den": maxes[idx]})
    g = d.groupby("vid").agg(num=("s", "sum"), den=("den", "sum"))
    return (g.num / g.den).to_dict()


def run(seed=42):
    D = build_design_matrix()
    X, y = D["X"], D["y"]
    binary, ordinal = config_columns()
    cfg_cols = [c for c in binary + ordinal if c in X.columns]

    with gzip.open(P.LONG_TABLE, "rt", encoding="utf-8") as fh:
        lg = pd.read_csv(fh, dtype={"variant_id": str})
    lg = lg[lg["role"] == "target"].copy()
    lg["max"] = pd.to_numeric(lg["max"], errors="coerce")
    lg["y"] = pd.to_numeric(lg["y"], errors="coerce")
    lg = lg[lg["y"].notna()].reset_index(drop=True)
    maxes = lg["max"].to_numpy()
    vids = lg["variant_id"].to_numpy()
    vers = lg["version"].to_numpy()
    allidx = np.arange(len(y))

    # 应用场景：用全部数据训练（非评估，故不切分）
    b2 = B2VehicleGBDT(seed)
    print(f"后端：{b2.backend}　配置维度 {len(cfg_cols)} 个\n")
    t0 = time.time()
    b2.b1.fit(X, y, _meta_slice(D, allidx))
    Xp = b2._prep(X, b2.b1.predict(X, _meta_slice(D, allidx)))
    pred_base, model = b2.fit_predict(Xp, y, Xp, [])
    pred_base = np.clip(pred_base, 0, 1)
    print(f"  基线训练与预测 {time.time()-t0:.0f}s")

    base_rate = module_rates(pred_base, vids, scns := lg["scenario"].to_numpy(), maxes, allidx)
    ver_of = dict(zip(vids, vers))

    # 逐项反事实：**复用同一个已训练模型**，只改输入重新预测。
    # 不可重新训练——重训会让模型适应新的特征分布，那就不是反事实而是另一个模型。
    cf_rate = {}
    for c in cfg_cols:
        Xc = Xp.copy()
        Xc[c] = 0.0                       # 二值置 0；序数降至最低档
        p = np.clip(model.predict(Xc), 0.0, 1.0)
        cf_rate[c] = module_rates(p, vids, scns, maxes, allidx)
        delta = float(np.mean(np.abs(p - pred_base)))
        print(f"  反事实 {c:30s} 细项预测平均变动 {delta:.4f}")

    # 该车原本是否配备了 c（用于只统计「有→无」的情形）
    has = {}
    vf = pd.read_csv(P.VEHICLE_FEAT, encoding="utf-8-sig", dtype={"variant_id": str})
    for c in cfg_cols:
        s = pd.to_numeric(vf.get(c), errors="coerce")
        has[c] = dict(zip(vf.variant_id, s > 0))

    rows = []
    for tname, thr in THRESHOLDS.items():
        for c in cfg_cols:
            per_ver = {}
            for vid, r0 in base_rate.items():
                if not has[c].get(vid, False):        # 原本就没配，无从消融
                    continue
                if r0 < thr:                          # 原本就不达标，不在讨论范围
                    continue
                r1 = cf_rate[c].get(vid)
                if r1 is None:
                    continue
                v = ver_of.get(vid)
                per_ver.setdefault(v, []).append((r1 < thr, r0 - r1))
            for v, lst in per_ver.items():
                if len(lst) < MIN_N:
                    continue
                nec = np.mean([a for a, _ in lst])
                drop = np.mean([b for _, b in lst])
                rows.append({"目标": tname, "版本": v, "配置": c, "n": len(lst),
                             "必要率": round(float(nec), 4),
                             "平均得分率损失": round(float(drop), 4)})
    d = pd.DataFrame(rows)
    d.to_csv(P.OUT / "e5_min_config.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 84)
    print("① 边际贡献：去掉该项配置后，乘员保护模块得分率的平均下降（主指标）")
    print("=" * 84)
    piv2 = d[d.目标 == "五星门槛"].pivot_table(index="配置", columns="版本",
                                             values="平均得分率损失")
    cols2 = [v for v in P.VORD if v in piv2.columns]
    piv2 = piv2[cols2]
    piv2["均值"] = piv2.mean(axis=1)
    print(piv2.sort_values("均值", ascending=False).round(4).to_string())

    print("\n" + "=" * 84)
    print("② 必要率：在已达标的车上去掉该项后，掉出门槛的比例（辅助指标）")
    print("=" * 84)
    for tname in THRESHOLDS:
        sub = d[d.目标 == tname]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="配置", columns="版本", values="必要率")
        cols = [v for v in P.VORD if v in piv.columns]
        piv = piv[cols]
        piv["最大"] = piv.max(axis=1)
        piv = piv.sort_values("最大", ascending=False)
        print(f"\n--- 目标：{tname}（模块得分率 ≥ {THRESHOLDS[tname]}）---")
        print(piv.round(3).to_string())

    print(f"\n→ {P.OUT / 'e5_min_config.csv'}")
    return d


if __name__ == "__main__":
    run()
