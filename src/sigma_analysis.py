# -*- coding: utf-8 -*-
"""
σ 变更矩阵与迁移折可比性核验（D-049 处置的验证环节）

回答三个问题：
    1. 相邻版本之间，哪些 item 的测量条件发生了变更？变更在哪一维？
    2. 每个前向迁移折的共有 item 子集中，comparable 与 incomparable 各占多少？
    3. F-19 断言的「F2 的 σ 全部未变、F3 的 σ 发生变更」是否成立？

可比性判据（写入本体 sigma_semantics）：
    同一 item_id 在版本 a 与 b 之间 comparable ⟺ (speed, barrier, dummy) 三维全同。
    任一维变更即 incomparable，**不做数值折算**——严酷度不作为可学习效应，
    只作为规则图上的标注（方案已确认，见核查终稿 §4）。

用法：  python src/sigma_analysis.py
"""
import json, sys, collections
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import paths as P


def load_item_sigma():
    """{item_id: {version: (speed, barrier, dummy)}}"""
    onto = json.loads(P.ONTO.read_text(encoding="utf-8"))
    out = {}
    for it in onto["items"]:
        per = {}
        for v, blk in (it.get("versions") or {}).items():
            s = blk.get("sigma") or {}
            per[v] = (s.get("speed_kmh", s.get("delta_v_kmh")),
                      s.get("barrier"), s.get("dummy"))
        out[it["item_id"]] = per
    return out


DIMS = ("speed", "barrier", "dummy")


def diff_sigma(a, b):
    """返回发生变更的维度名列表。"""
    return [d for d, x, y in zip(DIMS, a, b) if x != y]


def adjacent_change_matrix(sig):
    """相邻版本 σ 变更明细。"""
    rows = []
    for i in range(len(P.VORD) - 1):
        va, vb = P.VORD[i], P.VORD[i + 1]
        both = [k for k, per in sig.items() if va in per and vb in per]
        changed = collections.defaultdict(list)
        for k in both:
            d = diff_sigma(sig[k][va], sig[k][vb])
            for dim in d:
                changed[dim].append(k)
        n_incomp = len({k for ks in changed.values() for k in ks})
        rows.append({
            "transition": f"{va}→{vb}", "共有 item": len(both),
            "incomparable": n_incomp,
            "comparable": len(both) - n_incomp,
            "变更维": "，".join(f"{d}({len(v)})" for d, v in sorted(changed.items())) or "—",
        })
    return pd.DataFrame(rows)


def fold_comparability(sig, min_train_versions=2):
    """每个前向迁移折的共有 item 子集中 comparable / incomparable 构成。

    折的 σ 判据取「测试版本 vs 训练集中该 item 最近一次出现的版本」——
    这才是模型实际面对的条件差异；与最早版本比较会高估变更。
    """
    present = P.VORD
    rows, detail = [], {}
    for i in range(min_train_versions, len(present)):
        tr_v, te_v = present[:i], present[i]
        name = f"F{i - min_train_versions + 1}"
        shared, comp, incomp, dims = [], [], [], collections.Counter()
        for k, per in sig.items():
            if te_v not in per:
                continue
            prev = [v for v in tr_v if v in per]
            if not prev:
                continue                       # 训练期未出现 → 新 item，不在共有子集
            shared.append(k)
            last = prev[-1]
            d = diff_sigma(per[last], per[te_v])
            (incomp if d else comp).append(k)
            for x in d:
                dims[x] += 1
        rows.append({
            "fold": name, "train": "+".join(tr_v), "test": te_v,
            "共有 item": len(shared), "comparable": len(comp),
            "incomparable": len(incomp),
            "incomparable 率": round(len(incomp) / len(shared), 3) if shared else None,
            "变更维": "，".join(f"{d}({n})" for d, n in sorted(dims.items())) or "—",
        })
        detail[name] = {"comparable": sorted(comp), "incomparable": sorted(incomp)}
    return pd.DataFrame(rows), detail


def verify_f19(folds_df, detail):
    """复核 F-19：F2 的 σ 全部未变、F3 的 σ 发生变更。"""
    g = folds_df.set_index("fold")
    out = {}
    for f in ("F2", "F3"):
        if f not in g.index:
            out[f] = "该折不存在"
            continue
        r = g.loc[f]
        out[f] = {"共有 item": int(r["共有 item"]),
                  "incomparable": int(r["incomparable"]),
                  "变更维": r["变更维"],
                  "σ 全部未变": bool(r["incomparable"] == 0)}
    verdict = None
    if isinstance(out.get("F2"), dict) and isinstance(out.get("F3"), dict):
        ok = out["F2"]["σ 全部未变"] and not out["F3"]["σ 全部未变"]
        verdict = "✔ F-19 成立" if ok else "✘ F-19 与补全后的 σ 表不一致，须修订"
    return out, verdict


def main():
    sig = load_item_sigma()

    print("=== 1. 相邻版本 σ 变更矩阵 ===")
    adj = adjacent_change_matrix(sig)
    print(adj.to_string(index=False))

    print("\n=== 2. 前向迁移各折的可比性构成 ===")
    folds, detail = fold_comparability(sig)
    print(folds.to_string(index=False))

    print("\n=== 3. 复核 F-19（F2 σ 未变 / F3 σ 变更）===")
    res, verdict = verify_f19(folds, detail)
    for k, v in res.items():
        print(f"  {k}: {v}")
    print(f"\n  {verdict}")
    if "F3" in detail and detail["F3"]["incomparable"]:
        print("  F3 的 incomparable item（前 12 个）：")
        for k in detail["F3"]["incomparable"][:12]:
            print("    ", k)

    out = {"adjacent": adj.to_dict("records"), "folds": folds.to_dict("records"),
           "f19_verification": res, "f19_verdict": verdict, "fold_detail": detail}
    (P.OUT / "sigma_analysis.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    adj.to_csv(P.OUT / "sigma_adjacent_changes.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(P.OUT / "sigma_fold_comparability.csv", index=False, encoding="utf-8-sig")
    print(f"\n→ {P.OUT / 'sigma_analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
