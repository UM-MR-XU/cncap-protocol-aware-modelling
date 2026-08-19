# -*- coding: utf-8 -*-
"""
规则层 g_v（W2 交付物）

将细项得分向量映射为模块得分率与星级。全部参数来自规程原文（config/rule_config.json），
不含任何拟合量。版本切换只替换配置，不改代码。

    g_v : (细项得分, 车辆属性) ──► 试验项得分 ──► 模块得分率 ──► 判定量 ──► 星级

验收基线：571 台官方星级还原准确率 99.6%（W4 修正 V4 阈值后实测，未命中 2 例均为明火降星）

用法：  python src/rule_layer.py
"""
import csv, json, gzip, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths as P


# ══════════════════════════ 配置载入 ══════════════════════════
def load_config():
    return json.loads(P.RULE_CONFIG.read_text(encoding="utf-8"))


CFG = load_config()


def f(x):
    """宽松数值解析：容忍百分号、千分位与空占位符。"""
    if x is None:
        return None
    t = str(x).strip().replace(",", "").rstrip("%")
    if t in ("", "/", "-", "—", "N/A", "null", "None", "nan"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


# ══════════════════════════ 第一层：细项 → 试验项 ══════════════════════════
def aggregate_test_scores(item_scores):
    """item_scores: {item_id: score} → {scenario: 该试验各细项得分之和}

    注意：不含试验项级罚分（见决策日志 D-034）。官方试验项得分可能低于此值。
    """
    out = collections.defaultdict(float)
    for iid, sc in item_scores.items():
        if sc is None or not iid.startswith("occ."):
            continue
        parts = iid.split(".")
        if len(parts) < 3:
            continue
        out[parts[1]] += sc
    return dict(out)


# ══════════════════════════ 第二层：试验项 → 模块得分 ══════════════════════════
def occupant_score(test_scores, version, powertrain=None):
    """按版本公式汇总乘员保护部分实际得分。

    V6 两条互斥侧碰分支采用不同重标定系数（规程 2021 §4.1）：
        传统汽车 1.2 × 侧面碰撞；新能源汽车 1.5 × 侧面柱碰撞
    """
    g = lambda k: test_scores.get(k, 0.0)
    if version in ("V1", "V2", "V3", "V4", "V5"):
        return (g("frontal_frb100") + g("frontal_odb40") + g("side_mdb")
                + g("whiplash") + g("bonus") + g("penalty"))
    if version == "V6":
        base = g("frontal_frb100") + g("frontal_mpdb50") + g("child_static") + g("whiplash")
        if "side_pole" in test_scores:            # 新能源
            return base + 1.5 * g("side_pole")
        return base + 1.2 * g("side_mdb")
    if version == "V7":                            # 两项侧碰均须进行，不再分支
        return (g("frontal_frb100") + g("frontal_mpdb50") + g("side_mdb") + g("side_pole")
                + g("side_farside") + g("child_static") + g("whiplash")
                + g("cpd") + g("ecall") + g("curtain_hold"))
    raise ValueError(f"未知版本 {version}")


# ══════════════════════════ 第三层：判定量 → 星级 ══════════════════════════
def base_star(version, decision_value):
    """按阈值表给出基础星级。星级 6 表示五星+。"""
    for lower, star in CFG["star_thresholds"][version]:
        if lower is None or decision_value >= lower:
            return star
    return 1


ITEM_GATE_REGIONS_FRONTAL = {"head", "neck", "chest", "headneck"}
ITEM_GATE_REGIONS_SIDE = {"head", "chest", "abdomen", "pelvis"}
GATE_SCENARIOS = ("frontal_frb100", "frontal_odb40", "side_mdb")


def apply_item_gates(star, version, item_scores):
    """V3/V4 基于细项的降级。

    · 五星车：三项试验中前排假人特定部位不得为 0 分，否则降为四星
    · 四星车：每项试验前排假人得分不得低于 10 分，否则降为三星

    该规则以逐部位细项得分为输入——这正是「必须预测细项而非星级」的规程层面依据。
    """
    if version not in ("V3", "V4"):
        return star, []
    fired = []
    if star >= 5:
        for iid, sc in item_scores.items():
            p = iid.split(".")
            if len(p) < 4 or p[1] not in GATE_SCENARIOS or p[2] != "row1":
                continue
            allowed = ITEM_GATE_REGIONS_SIDE if p[1] == "side_mdb" else ITEM_GATE_REGIONS_FRONTAL
            if p[3] in allowed and sc is not None and abs(sc) < 1e-9:
                fired.append(f"zero_part:{iid}")
        if fired:
            star = 4
    if star == 4:
        per_test = collections.defaultdict(float)
        for iid, sc in item_scores.items():
            p = iid.split(".")
            if len(p) >= 4 and p[1] in GATE_SCENARIOS and p[2] == "row1" and sc is not None:
                per_test[p[1]] += sc
        low = [f"row1_sum<10:{k}={v:.2f}" for k, v in per_test.items() if v < 10 - 1e-9]
        if low:
            fired += low
            star = 3
    return star, fired


def apply_module_minimums(star, version, occupant_rate, vru_rate, active_rate, test_year):
    """V5 起：不满足各部分最低得分率则按其得分率达到的最低星级评定。"""
    table = CFG["module_min_rate"].get(version)
    if not table:
        return star, []
    fired = []
    s = star
    while s > 1:
        req = table.get(str(s))
        if req is None:
            break
        ok = True
        if occupant_rate is not None and "occupant" in req:
            ok &= occupant_rate >= req["occupant"] - 1e-9
        if vru_rate is not None and "vru" in req:
            ok &= vru_rate >= req["vru"] - 1e-9
        act = req.get("active") if "active" in req else None
        if act is None and "active_by_test_year" in req:
            m = req["active_by_test_year"]
            act = m.get(str(test_year), m.get("_default"))
        if active_rate is not None and act is not None:
            ok &= active_rate >= act - 1e-9
        if ok:
            break
        s -= 1
    if s != star:
        fired.append(f"module_min:{star}->{s}")
    return s, fired


def decide_star(version, decision_value, *, occupant_rate=None, vru_rate=None,
                active_rate=None, test_year=None, item_scores=None):
    """完整的 g_v 星级判定链。返回 (星级, 触发的规则列表)。"""
    star = base_star(version, decision_value)
    trace = [f"base:{star}"]
    if item_scores:
        star, fired = apply_item_gates(star, version, item_scores)
        trace += fired
    star, fired = apply_module_minimums(star, version, occupant_rate, vru_rate,
                                        active_rate, test_year)
    trace += fired
    return star, trace


# ══════════════════════════ 验收测试 ══════════════════════════
def read_csv(path, enc="utf-8-sig"):
    with open(path, encoding=enc) as fh:
        return list(csv.DictReader(fh))


def run_validation():
    # Public release: official outcomes come from the de-identified table in
    # data/. The original working tree joined two raw scrape files here; the
    # joined result is what is shipped, so no join is needed.
    lst = {r["variant_id"]: r for r in read_csv(P.OFFICIAL)}

    with gzip.open(P.LONG_TABLE, "rt", encoding="utf-8") as fh:
        long_rows = list(csv.DictReader(fh))
    scores = collections.defaultdict(dict)
    for r in long_rows:
        scores[r["variant_id"]][r["item_id"]] = f(r["score"])

    pct = lambda x: (x / 100 if x is not None and x > 1 else x)
    res_base = collections.Counter()
    res_full = collections.Counter()
    misses = []

    for rid, r in lst.items():
        v = r["version"]
        if v not in P.VORD:
            continue
        official = f(r["star_rating"])
        raw = f(r.get("overall_score"))
        if official is None or raw is None:
            continue
        dv = raw if v in ("V1", "V2", "V3", "V4") else pct(raw)
        b = base_star(v, dv)
        s, trace = decide_star(
            v, dv,
            occupant_rate=pct(f(r.get("ap_score"))),
            vru_rate=pct(f(r.get("vru_score"))),
            active_rate=pct(f(r.get("as_score"))),
            test_year=r.get("test_year"),
            item_scores=scores.get(rid),
        )
        res_base[(v, b == official)] += 1
        res_full[(v, s == official)] += 1
        if s != official:
            misses.append({"variant_id": rid, "version": v, "decision_value": dv,
                           "predicted": s, "official": official, "trace": trace})

    print(f'{"版本":<7}{"仅阈值":>9}{"完整规则层":>12}{"n":>6}')
    A = B = N = 0
    for v in P.VORD:
        n = res_base[(v, True)] + res_base[(v, False)]
        if not n:
            continue
        a, b2 = res_base[(v, True)] / n, res_full[(v, True)] / n
        print(f"{v:<7}{a:>8.1%}{b2:>11.1%}{n:>6}")
        A += res_base[(v, True)]; B += res_full[(v, True)]; N += n
    print(f'{"合计":<7}{A/N:>8.1%}{B/N:>11.1%}{N:>6}')

    base_line = CFG["validation_baseline"]["star_recovery_accuracy"]["overall"]
    acc = B / N
    ok = abs(acc - base_line) <= 0.002
    print(f"\n验收基线 {base_line:.1%}｜实测 {acc:.1%}｜{'通过 ✔' if ok else '未通过 ✘'}")

    by_v = collections.Counter(m["version"] for m in misses)
    print(f"未命中 {len(misses)} 例，按版本：{dict(sorted(by_v.items()))}")
    print("（V4 未命中系「碰撞后 3 分钟内明火降一星」——现场观测事件，数据中无对应字段）")

    (P.OUT / "rule_layer_validation.json").write_text(json.dumps(
        {"overall_accuracy": acc, "baseline": base_line, "passed": ok,
         "n": N, "misses": misses[:60],
         "by_version": {v: {"threshold_only": res_base[(v, True)] / max(res_base[(v, True)] + res_base[(v, False)], 1),
                            "full": res_full[(v, True)] / max(res_full[(v, True)] + res_full[(v, False)], 1),
                            "n": res_base[(v, True)] + res_base[(v, False)]}
                        for v in P.VORD if res_base[(v, True)] + res_base[(v, False)]}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run_validation())
