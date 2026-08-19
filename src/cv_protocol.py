# -*- coding: utf-8 -*-
"""
交叉验证协议与评估（W4 交付物）

两套互不替代的切分：

    A. 分组 K 折（GroupKFold by variant_id）
       同一车辆的全部细项必须落在同一折。同车不同细项高度相关，
       随机按行切分会造成严重泄漏——这是本数据集最容易犯的错误。
       用途：模型选择、超参调优、消融。

    B. 滚动原点前向迁移（rolling-origin forward transfer）
       F_k：训练 V1..V_k，测试 V_{k+1}，k = 2..6，共 5 折。
       只用过去预测未来，与规程实际演进顺序一致。
       用途：跨版本迁移能力的主实验。

评估一律**分层报告**，不只报总体：
    · 按版本、按工况、按可观测性类别
    · 每一层都并列给出该层的平凡基线（恒 1.0 与该层均值）

    ⚠ 不给平凡基线的误差数字没有意义——全库 61% 的标签为 1.0，
      恒输出 1.0 的 MAE 已是 0.1371（F-35）。

用法：  python src/cv_protocol.py     # 自检：切分无泄漏、覆盖完整
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paths as P


# ══════════════════════ A. 分组 K 折 ══════════════════════
def group_kfold_indices(groups, n_splits=5, seed=42):
    """按 variant_id 分组的 K 折，返回 [(train_idx, test_idx), ...]。

    不依赖 sklearn。将车辆随机打散后均分到 K 折——按车辆数而非行数均分，
    因各车行数不等（不同版本细项数不同），按行均分会打破分组。
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    assign = np.array([fold_of[g] for g in groups])
    out = []
    for k in range(n_splits):
        te = np.where(assign == k)[0]
        tr = np.where(assign != k)[0]
        assert not (set(groups[tr]) & set(groups[te])), "分组泄漏：同一车辆同时出现在训练与测试"
        out.append((tr, te))
    return out


# ══════════════════════ B. 滚动原点前向迁移 ══════════════════════
def forward_transfer_folds(version, min_train_versions=2):
    """F_k：训练 V1..V_k → 测试 V_{k+1}。返回 [(name, train_idx, test_idx, 训练版本, 测试版本), ...]"""
    version = np.asarray(version)
    present = [v for v in P.VORD if (version == v).any()]
    out = []
    for i in range(min_train_versions, len(present)):
        tr_v, te_v = present[:i], present[i]
        tr = np.where(np.isin(version, tr_v))[0]
        te = np.where(version == te_v)[0]
        if len(tr) and len(te):
            out.append((f"F{i - min_train_versions + 1}", tr, te, tr_v, te_v))
    return out


def shared_item_subset(item_id, train_idx, test_idx):
    """训练集与测试集共有的 item 子集索引（D-031）。

    F2/F3 关键对照必须在共有子集上做——原设计依赖「两折 Jaccard 恰为 1.00」
    这一巧合，伪数据剔除后该条件不再成立，故改为显式取交集。
    """
    item_id = np.asarray(item_id)
    shared = set(item_id[train_idx]) & set(item_id[test_idx])
    mask = np.isin(item_id, list(shared))
    return np.where(mask[test_idx])[0], sorted(shared)


# ══════════════════════ 评估 ══════════════════════
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = y_true - y_pred
    sub = y_true < 1 - 1e-9
    return {
        "n": int(len(y_true)),
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        # 平凡基线：无任何特征即可达到的水平，任何模型须显著优于此
        "MAE_const1": float(np.abs(y_true - 1.0).mean()),
        "RMSE_const1": float(np.sqrt(((y_true - 1.0) ** 2).mean())),
        "MAE_mean": float(np.abs(y_true - y_true.mean()).mean()),
        "RMSE_mean": float(np.sqrt(((y_true - y_true.mean()) ** 2).mean())),
        # 非满分子集——区分模型能力的敏感区间（F-35）
        "MAE_partial": float(np.abs(e[sub]).mean()) if sub.any() else float("nan"),
        "n_partial": int(sub.sum()),
        "full_score_rate": float(np.mean(~sub)),
    }


def skill_score(m, key="MAE"):
    """相对最强平凡基线的技能得分：1 - 模型误差 / 平凡误差。>0 才算有价值。"""
    base = min(m[f"{key}_const1"], m[f"{key}_mean"])
    return float("nan") if base <= 0 else 1.0 - m[key] / base


def evaluate(y_true, y_pred, *, by=None, name="overall"):
    """总体 + 分层评估，返回 DataFrame。by 为 {层名: 分组数组} 的 dict。"""
    rows = [{"stratum": "总体", "level": name, **metrics(y_true, y_pred)}]
    for lname, arr in (by or {}).items():
        arr = np.asarray(arr)
        for lv in pd.unique(arr):
            m = arr == lv
            if m.sum() >= 5:
                rows.append({"stratum": lname, "level": str(lv),
                             **metrics(y_true[m], y_pred[m])})
    df = pd.DataFrame(rows)
    df["skill_MAE"] = df.apply(lambda r: skill_score(r, "MAE"), axis=1)
    df["skill_RMSE"] = df.apply(lambda r: skill_score(r, "RMSE"), axis=1)
    return df


def report(df, digits=4):
    cols = ["stratum", "level", "n", "MAE", "RMSE", "MAE_const1", "RMSE_mean",
            "skill_MAE", "skill_RMSE", "MAE_partial", "full_score_rate"]
    return df[[c for c in cols if c in df.columns]].round(digits).to_string(index=False)


# ══════════════════════ 自检 ══════════════════════
def _selftest():
    from dataset import build_design_matrix
    D = build_design_matrix()
    y, g, v, iid = D["y"], D["groups"], D["version"], D["item_id"]

    print("=== A. 分组 5 折（by variant_id）===")
    folds = group_kfold_indices(g, 5)
    cover = np.zeros(len(y), bool)
    for k, (tr, te) in enumerate(folds):
        cover[te] = True
        print(f"  折{k}｜训练 {len(tr):5d} 行 / {len(set(g[tr])):3d} 车"
              f"｜测试 {len(te):5d} 行 / {len(set(g[te])):3d} 车")
    assert cover.all(), "分组 K 折未覆盖全部样本"
    print("  ✔ 无同车泄漏，覆盖完整")

    print("\n=== B. 滚动原点前向迁移 ===")
    for nm, tr, te, tv, tev in forward_transfer_folds(v):
        sub, shared = shared_item_subset(iid, tr, te)
        n_new = len(set(iid[te]) - set(iid[tr]))
        print(f"  {nm}｜训练 {'+'.join(tv)} ({len(tr):5d} 行) → 测试 {tev} ({len(te):4d} 行)"
              f"｜共有 item {len(shared):2d}｜测试集中的新 item {n_new:2d}"
              f"｜共有子集覆盖测试集 {len(sub) / len(te):.0%}")
    print("  ⚠ 「新 item」= 训练期从未出现的细项。宽表多任务架构对其无输出头，")
    print("    长表编码可据本体三元组外推——这正是 F-21 的实验体现。")

    print("\n=== C. 平凡基线（全库）===")
    print(report(evaluate(y, np.full_like(y, 1.0),
                          by={"版本": v, "工况": D["scenario"],
                              "可观测性": D["observability"]},
                          name="恒输出 1.0")))


if __name__ == "__main__":
    _selftest()
