# -*- coding: utf-8 -*-
"""
细项 → 试验项 汇总链路验证（W2）

检验规则层第一层的正确性：Σ(细项得分) 是否等于官方试验项得分。
差值即为**试验项级罚分**——它不体现在任何细项中，仅以页面注释文本给出（见 D-034）。

用法：  python src/validate_aggregation.py
"""
import csv, json, glob, gzip, os, re, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths as P
from build_long_table import read_csv, fnum, record_id, load_ontology


def official_test_scores():
    """从原始模块 JSON 提取官方试验项得分：{(variant_id, scenario): (score, denom)}"""
    _, _, mapping, _ = load_ontology()
    # 由映射表反推 原文试验项名 → scenario
    test2scn = {}
    for (v, rt, rd, rr), iid in mapping.items():
        if iid.startswith("occ."):
            test2scn[(v, rt)] = iid.split(".")[1]

    lst = {r["id"]: r for r in read_csv(P.LIST_CSV)}
    det = {r["detail_url"]: r for r in read_csv(P.DETAIL_CSV)}
    out = {}
    for pattern in P.MODULE_GLOBS:
        for fp in glob.glob(str(pattern)):
            rid = record_id(fp)
            r = lst.get(rid)
            d = det.get(r["detail_url"]) if r else None
            if not d:
                continue
            v = P.VERSION_MAP.get(d["regulation_version"])
            if not v or (v, rid) in P.BAD_RECORDS:
                continue
            occ = (json.loads(Path(fp).read_text(encoding="utf-8")) or {}).get("乘员保护模块") or {}
            blocks = list(occ.get("乘员保护试验列表") or [])
            for k, val in occ.items():
                if isinstance(val, dict) and "试验得分" in val:
                    blocks.append({**val, "试验项": val.get("试验项") or k})
            for t in blocks:
                tn, s = t.get("试验项"), str(t.get("试验得分") or "")
                scn = test2scn.get((v, tn))
                if not scn or "/" not in s:
                    continue
                try:
                    num, den = s.split("/")
                    out[(rid, scn)] = (float(num), float(den), v)
                except ValueError:
                    continue
    return out


def main():
    with gzip.open(P.LONG_TABLE, "rt", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    item_sum = collections.defaultdict(float)
    for r in rows:
        if r["role"] not in ("target", "aggregation_source"):
            continue
        sc = fnum(r["score"])
        if sc is not None:
            item_sum[(r["variant_id"], r["scenario"])] += sc

    official = official_test_scores()
    diffs = collections.defaultdict(list)
    detail = []
    for key, (off, den, v) in official.items():
        if key not in item_sum:
            continue
        got = item_sum[key]
        d = round(got - off, 3)
        diffs[(v, key[1])].append(d)
        if abs(d) > 1e-3:
            detail.append({"variant_id": key[0], "version": v, "scenario": key[1],
                           "item_sum": round(got, 3), "official": off, "gap": d})

    print(f'{"版本":<5}{"工况":<20}{"可比车数":>7}{"完全一致":>8}{"有差值":>7}{"差值中位":>9}{"最大差值":>9}')
    tot = ok = 0
    for (v, scn), ds in sorted(diffs.items()):
        n = len(ds)
        z = sum(1 for x in ds if abs(x) <= 1e-3)
        nz = [x for x in ds if abs(x) > 1e-3]
        med = sorted(nz)[len(nz) // 2] if nz else 0.0
        mx = max(nz, key=abs) if nz else 0.0
        tot += n; ok += z
        print(f"{v:<5}{scn:<20}{n:>7}{z:>8}{len(nz):>7}{med:>9.3f}{mx:>9.3f}")
    print(f"\n合计可比 {tot} 组，完全一致 {ok}（{ok/tot:.1%}），有差值 {tot-ok}")

    # 差值方向分析
    pos = sum(1 for d in detail if d["gap"] > 0)
    print(f"其中 细项和 > 官方（即存在试验项级罚分）: {pos} 组；细项和 < 官方: {len(detail)-pos} 组")
    print("\n差值最大的 8 组：")
    for d in sorted(detail, key=lambda x: -abs(x["gap"]))[:8]:
        print(f"   {d['version']} id={d['variant_id']:>3} {d['scenario']:<18} "
              f"细项和 {d['item_sum']:>7.3f}  官方 {d['official']:>7.3f}  差 {d['gap']:>+7.3f}")

    (P.OUT / "aggregation_validation.json").write_text(json.dumps(
        {"n_comparable": tot, "n_exact": ok, "exact_rate": ok / tot,
         "n_with_gap": len(detail), "gaps": detail[:200],
         "by_version_scenario": {f"{v}|{s}": {"n": len(ds),
                                              "exact": sum(1 for x in ds if abs(x) <= 1e-3)}
                                 for (v, s), ds in diffs.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    P.require_raw("validate_aggregation.py")
    main()
