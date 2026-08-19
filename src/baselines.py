# -*- coding: utf-8 -*-
"""
基线模型 B0 / B1 / B2（W4 交付物）

三条基线的作用是**划定「不学习」与「学习」的分界**，读者据此判断主模型的增益来自何处：

    B0  全局常数         无任何信息。恒 1.0 与全局均值两种，MAE/RMSE 下排序相反（F-35）。
    B1  item 层级均值    只用「这是哪个细项」，不用任何车辆信息。
                        含层级回退：item → (scenario,region) → scenario → 全局，
                        使其能对训练期未见的新 item 给出预测——这是长表编码思想的
                        最简实现，也是 B2 必须超越的对象。
    B2  item 均值 + 车辆特征的 GBDT
                        在 B1 之上加入车辆特征。B2 − B1 即**车辆信息的净增益**。

⚠ 若 B2 相对 B1 的提升很小，说明模型主要在记忆细项难度而非利用车辆设计信息——
   这是本类工作最可能出现的负面结果，须如实报告，不可只报 B2 对 B0 的提升。

环境
----
B0/B1 仅需 numpy/pandas。B2 需要 lightgbm（推荐）或 sklearn，二者皆无时自动跳过
并提示——沙箱内 pip 受限，B2 请在本地环境运行。

用法
----
    python src/baselines.py                 # 分组 5 折 + 前向迁移，全部基线
    python src/baselines.py --cv group      # 只跑分组 K 折
    python src/baselines.py --cv forward    # 只跑前向迁移
    python src/baselines.py --component     # 加入部件级族系代理特征
"""
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paths as P
from dataset import build_design_matrix, summarize
from cv_protocol import (group_kfold_indices, forward_transfer_folds,
                         shared_item_subset, evaluate, report)


# ══════════════════════ B0：全局常数 ══════════════════════
class B0Constant:
    """mode='mean' 用训练集均值；mode='one' 恒输出 1.0（= 中位数）。"""

    def __init__(self, mode="mean"):
        self.mode, self.c = mode, 1.0

    def fit(self, X, y, meta):
        self.c = 1.0 if self.mode == "one" else float(np.mean(y))
        return self

    def predict(self, X, meta):
        return np.full(len(meta["item_id"]), self.c, dtype=float)


# ══════════════════════ B1：item 层级均值 ══════════════════════
class B1ItemMean:
    """按 item_id 取训练集均值，对未见 item 逐级回退。

    回退链 item_id → (scenario,region) → scenario → 全局，
    使基线具备对新细项的外推能力，从而与主模型可比。
    """

    def fit(self, X, y, meta):
        d = pd.DataFrame({"item": meta["item_id"], "scn": meta["scenario"],
                          "reg": _region_of(meta["item_id"]), "y": y})
        self.by_item = d.groupby("item").y.mean().to_dict()
        self.by_sr = d.groupby(["scn", "reg"]).y.mean().to_dict()
        self.by_scn = d.groupby("scn").y.mean().to_dict()
        self.glob = float(d.y.mean())
        return self

    def predict(self, X, meta):
        item, scn = meta["item_id"], meta["scenario"]
        reg = _region_of(item)
        out = np.empty(len(item), dtype=float)
        for i in range(len(item)):
            out[i] = self.by_item.get(
                item[i], self.by_sr.get((scn[i], reg[i]),
                                        self.by_scn.get(scn[i], self.glob)))
        return out


def _region_of(item_id):
    return np.array([s.split(".")[-1] if isinstance(s, str) else "" for s in item_id])


# ══════════════════════ B2：item 均值 + 车辆特征 GBDT ══════════════════════
def monotone_vector(columns, enable=True):
    """按 feature_groups.json 生成单调约束向量（E3 消融用）。

    只对 `passive_safety` 组的二值与序数特征施加 +1（装配率提升 ⇒ 得分不降）。
    不约束的三类及其理由：
      · 整车参数——质量与尺寸对伤害的作用方向随工况而异，无统一先验
      · σ 特征——速度提升应使得分下降，但车辆同期在进步，观测方向不纯（F-37）
      · 本体三元组——无序类别，单调无意义
    """
    if not enable:
        return None
    fg = json.loads(P.FEATURE_GROUPS.read_text(encoding="utf-8"))
    g = fg.get("passive_safety", {})
    mono_cols = set(g.get("binary", []) + g.get("ordinal", []))
    return [1 if c in mono_cols else 0 for c in columns]


def _get_gbdt(seed=42, monotone=None):
    """返回 (拟合函数, 后端名)。

    优先级：lightgbm > sklearn > 自实现。前二者为成熟实现，本地环境应优先；
    自实现（`src/gbdt.py`）用于 pip 不可用的环境，并作为交叉验证对照。
    三者均原生支持缺失值与单调约束。
    """
    try:
        import lightgbm as lgb

        def fit_predict(Xtr, ytr, Xte, cat_cols):
            kw = {}
            if monotone is not None:
                kw["monotone_constraints"] = monotone
            m = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                verbose=-1, **kw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Xtr, ytr, categorical_feature=cat_cols)
                return m.predict(Xte), m
        return fit_predict, "lightgbm"
    except ImportError:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor

        def fit_predict(Xtr, ytr, Xte, cat_cols):
            kw = {}
            if monotone is not None:
                kw["monotonic_cst"] = monotone
            m = HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                min_samples_leaf=20, l2_regularization=1.0, random_state=seed,
                categorical_features=[c in cat_cols for c in Xtr.columns], **kw)
            m.fit(Xtr, ytr)
            return m.predict(Xte), m
        return fit_predict, "sklearn-HistGBR"
    except ImportError:
        pass
    from gbdt import GBDTRegressor

    def fit_predict(Xtr, ytr, Xte, cat_cols):
        m = GBDTRegressor(n_estimators=300, learning_rate=0.06, max_depth=6,
                          min_samples_leaf=20, l2=1.0, subsample=0.8,
                          colsample=0.8, monotone_constraints=monotone,
                          random_state=seed)
        m.fit(Xtr.to_numpy(dtype=float), ytr)
        return m.predict(Xte.to_numpy(dtype=float)), m
    return fit_predict, "numpy-GBDT（自实现）"


class B2VehicleGBDT:
    """B1 的预测值作为一列特征输入 GBDT，其余为车辆特征与本体三元组。

    以 B1 预测作特征（而非残差建模）的理由：让树自行决定何时依赖细项先验、
    何时依赖车辆信息，避免人为设定残差尺度。
    """

    def __init__(self, seed=42):
        self.fit_predict, self.backend = _get_gbdt(seed)
        self.b1 = B1ItemMean()

    def available(self):
        return self.fit_predict is not None

    def _prep(self, X, b1pred):
        """注意：本方法在设计矩阵末尾追加 b1_item_prior 一列。
        任何按列构造的向量（如单调约束）必须同步扩展，否则长度不匹配。"""
        d = X.copy()
        d["b1_item_prior"] = b1pred
        for c in d.columns:
            if str(d[c].dtype) == "category":
                d[c] = d[c].cat.codes.replace(-1, np.nan)
        return d

    def fit_predict_fold(self, X, y, tr, te, meta_tr, meta_te, cat_cols):
        self.b1.fit(X.iloc[tr], y[tr], meta_tr)
        Xtr = self._prep(X.iloc[tr], self.b1.predict(X.iloc[tr], meta_tr))
        Xte = self._prep(X.iloc[te], self.b1.predict(X.iloc[te], meta_te))
        pred, model = self.fit_predict(Xtr, y[tr], Xte, [])
        return np.clip(pred, 0.0, 1.0), model      # 标签定义域为 [0,1]，越界无意义


# ══════════════════════ 运行 ══════════════════════
def _meta_slice(D, idx):
    return {k: np.asarray(D[k])[idx] for k in ("item_id", "scenario", "observability", "version")}


def run_group_cv(D, n_splits=5, seed=42):
    X, y = D["X"], D["y"]
    folds = group_kfold_indices(D["groups"], n_splits, seed)
    b2 = B2VehicleGBDT(seed)
    preds = {"B0_mean": np.zeros(len(y)), "B0_one": np.ones(len(y)),
             "B1_item": np.zeros(len(y))}
    if b2.available():
        preds["B2_gbdt"] = np.zeros(len(y))

    for tr, te in folds:
        mtr, mte = _meta_slice(D, tr), _meta_slice(D, te)
        preds["B0_mean"][te] = B0Constant("mean").fit(X.iloc[tr], y[tr], mtr).predict(X.iloc[te], mte)
        preds["B1_item"][te] = B1ItemMean().fit(X.iloc[tr], y[tr], mtr).predict(X.iloc[te], mte)
        if b2.available():
            preds["B2_gbdt"][te], _ = b2.fit_predict_fold(X, y, tr, te, mtr, mte, D["categorical_cols"])

    out = {}
    for name, p in preds.items():
        out[name] = evaluate(y, p, name=name, by={
            "版本": D["version"], "工况": D["scenario"], "可观测性": D["observability"]})
    return out, b2.backend


def run_forward_transfer(D, seed=42):
    X, y, iid = D["X"], D["y"], D["item_id"]
    b2 = B2VehicleGBDT(seed)
    rows = []
    for nm, tr, te, tv, tev in forward_transfer_folds(D["version"]):
        mtr, mte = _meta_slice(D, tr), _meta_slice(D, te)
        sub, shared = shared_item_subset(iid, tr, te)
        n_new = len(set(iid[te]) - set(iid[tr]))
        cand = {"B0_mean": B0Constant("mean").fit(X.iloc[tr], y[tr], mtr).predict(X.iloc[te], mte),
                "B1_item": B1ItemMean().fit(X.iloc[tr], y[tr], mtr).predict(X.iloc[te], mte)}
        if b2.available():
            cand["B2_gbdt"], _ = b2.fit_predict_fold(X, y, tr, te, mtr, mte, D["categorical_cols"])
        for name, p in cand.items():
            for scope, idx in (("全部 item", slice(None)), ("共有 item 子集", sub)):
                m = evaluate(y[te][idx], p[idx], name=name).iloc[0].to_dict()
                rows.append({"fold": nm, "train": "+".join(tv), "test": tev,
                             "model": name, "scope": scope,
                             "n_shared_item": len(shared), "n_new_item": n_new, **m})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", choices=["group", "forward", "both"], default="both")
    ap.add_argument("--component", action="store_true", help="加入部件级族系代理（仅 V3–V5 可得）")
    ap.add_argument("--drop-config-gated", action="store_true", help="剔除 config_gated 类细项")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    D = build_design_matrix(include_component=a.component,
                            drop_config_gated=a.drop_config_gated)
    print("=== 数据集 ===");  print(summarize(D));  print()

    backend = _get_gbdt()[1]
    if backend is None:
        print("⚠ 未检测到 lightgbm 或 sklearn，B2 将跳过。")
        print("  B0/B1 结果仍然有效且已足以确立平凡基线。")
        print("  安装：pip install lightgbm    然后重跑本脚本以获得 B2。\n")
    else:
        print(f"GBDT 后端：{backend}\n")

    outdir = P.OUT
    if a.cv in ("group", "both"):
        res, _ = run_group_cv(D, seed=a.seed)
        print("=== A. 分组 5 折（by variant_id）===")
        for name, df in res.items():
            print(f"\n--- {name} ---")
            print(report(df[df.stratum.isin(["总体", "工况"])]))
            df.insert(0, "model", name)
        allres = pd.concat(res.values(), ignore_index=True)
        allres.to_csv(outdir / "baseline_group_cv.csv", index=False, encoding="utf-8-sig")
        print(f"\n→ {outdir / 'baseline_group_cv.csv'}")

    if a.cv in ("forward", "both"):
        ft = run_forward_transfer(D, seed=a.seed)
        print("\n=== B. 滚动原点前向迁移 ===")
        cols = ["fold", "train", "test", "model", "scope", "n", "n_new_item",
                "MAE", "RMSE", "MAE_const1", "skill_MAE", "full_score_rate"]
        print(ft[cols].round(4).to_string(index=False))
        ft.to_csv(outdir / "baseline_forward_transfer.csv", index=False, encoding="utf-8-sig")
        print(f"\n→ {outdir / 'baseline_forward_transfer.csv'}")

    print("\n判读要点：")
    print("  1. skill_MAE ≤ 0 表示不优于平凡基线。")
    print("  2. B2 − B1 才是车辆特征的净增益；B2 − B0 会把细项难度先验算作模型功劳。")
    print("  3. 前向迁移须同时看「全部 item」与「共有 item 子集」——前者含新细项外推，")
    print("     后者是拓扑不变条件下的纯 σ／分布漂移效应（D-031）。")


if __name__ == "__main__":
    main()
