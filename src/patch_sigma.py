# -*- coding: utf-8 -*-
"""
σ 表补全与下发（D-049 处置）

背景
----
`item_ontology_v1.json` 的 `sigma_table` 原本只有 `side_mdb` 填了 `barrier`，
正面三工况、侧面柱碰、鞭打均无壁障项；部分工况亦缺假人项。σ 是 comparable /
incomparable 判定的依据，缺项使跨版本可比性标注建立在不完整信息上。

本脚本做两件事，幂等、可重复运行：
    1. 用下方 SIGMA（全部逐条注明规程出处）覆盖 `sigma_table`
    2. 按 scenario 下发到每个 `items[].versions[].sigma`

σ 的三个取值语义（必须区分，不可混用 null）
-------------------------------------------
    实际值          该维度存在且已从规程抄录
    "N/A-STATIC"    该工况存在但无此维度（静态评价无碰撞速度／壁障）
    键不存在        该工况在该版本不存在

用法：  python src/patch_sigma.py [--dry-run]
"""
import argparse, json, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths as P

NA = "N/A-STATIC"
V17 = ["V1", "V2", "V3", "V4", "V5", "V6", "V7"]


def _spread(value, versions):
    return {v: value for v in versions}


# ══════════════════════════════════════════════════════════════════
# σ 参数表 —— 每条注明规程出处。修改此表前必须回查原文。
# ══════════════════════════════════════════════════════════════════
SIGMA = {
    # ── 正面 100% 重叠刚性壁障 ──────────────────────────────
    # 2006 §? 50⁺¹₀；2024 正文 3.1.1.1 明载「由 2021 版的 50⁺¹₀ 提高到 55⁺¹₀ km/h」
    "frontal_frb100": {
        "speed_kmh": {**_spread(50, ["V1", "V2", "V3", "V4", "V5", "V6"]), "V7": 55},
        # 固定刚性壁障，七版均为钢筋混凝土固定壁障，型式未变（2024 附录 A.10.5）
        "barrier": _spread("RB-FIXED", V17),
        "row1_dummy": _spread("HIII-50th", V17),
        # 2006 起第二排即放 HIII 5th 女性假人（2006 规程；早期仅考核安全带，不评分）
        "row2_dummy": _spread("HIII-5th", V17),
        # 2009 起第二排另一侧增加儿童假人。P 系列 → Q 系列的切换时点：
        #   V2 2009「增加 P 系列 3 岁儿童假人」（changelog 原文）
        #   V3 2012「P 系列 3 岁儿童假人」（原文，全文 5 处，无 Q 系列）
        #   V4 2015 **待核**——扫描版未查，暂沿用 P3（见 T-11）
        #   V5 2018「Q 系列 3 岁儿童假人」（首发版与修订版原文逐字相同）
        # ⚠ V2–V5 的儿童假人规程表述为「用以**考核**乘员约束系统性能」而非「用以**测量**
        #   受伤害情况」，按 D-011 判据不参与评分，本体中无对应 item（全部 row2_child
        #   细项 present_in 均为 V6/V7）。故本行取值不影响任何计算，仅为本体完整性。
        "row2_child_dummy": {**_spread("P3", ["V2", "V3", "V4"]),
                             **_spread("Q3", ["V5", "V6", "V7"])},
    },

    # ── 正面 40% 重叠可变形壁障（V1–V5，V6 起被 MPDB 取代）────
    # 2006/2009 为 56⁺¹₀；2012 起提至 64⁺¹₀
    "frontal_odb40": {
        "speed_kmh": {"V1": 56, "V2": 56, "V3": 64, "V4": 64, "V5": 64},
        "barrier": _spread("ODB", ["V1", "V2", "V3", "V4", "V5"]),
        "row1_dummy": _spread("HIII-50th", ["V1", "V2", "V3", "V4", "V5"]),
        "row2_dummy": _spread("HIII-5th", ["V1", "V2", "V3", "V4", "V5"]),
    },

    # ── 正面 50% 重叠移动渐进变形壁障（V6 起）─────────────────
    # 2021 正文 3.1.1.2 与 2024 正文 3.1.1.2 均为 50⁺¹₋₁ km/h（试验车与台车对撞）
    # 驾驶员位 THOR 50th，前排乘员位 Hybrid III 5th 女性——与 FRB 的前排配置不同
    "frontal_mpdb50": {
        "speed_kmh": {"V6": 50, "V7": 50},
        "barrier": {"V6": "MPDB", "V7": "MPDB"},
        "row1_dummy": {"V6": "THOR-50th+HIII-5th", "V7": "THOR-50th+HIII-5th"},
        "row2_dummy": {"V6": "HIII-5th", "V7": "HIII-5th"},
        "row2_child_dummy": {"V6": "Q10", "V7": "Q10"},
    },

    # ── 可变形移动壁障侧面碰撞 ────────────────────────────────
    # 2021 正文 3.1.1.3 为 50⁺¹₀；2024 正文 3.1.1.3 为 60⁺¹₀，
    # 且 2024 正文明载「移动壁障前端蜂窝铝型号变更」→ AE-MDB 改为 SC-MDB
    # 前排假人 2018 起由 EuroSID-II 改为 WorldSID-50th（核查终稿 §4，已由 2006/2009 原文确认沿用）
    "side_mdb": {
        "speed_kmh": {**_spread(50, ["V1", "V2", "V3", "V4", "V5", "V6"]), "V7": 60},
        "barrier": {**_spread("AE-MDB", ["V1", "V2", "V3", "V4", "V5", "V6"]), "V7": "SC-MDB"},
        "row1_dummy": {**_spread("EuroSID-II", ["V1", "V2", "V3", "V4"]),
                       **_spread("WorldSID-50th", ["V5", "V6", "V7"])},
        # 2009 起第二排撞击侧增加 SID-IIs（D 版），但 V2–V4 不评分
        "row2_dummy": _spread("SID-IIs", ["V2", "V3", "V4", "V5", "V6", "V7"]),
    },

    # ── 侧面柱碰撞（V6 起；2021 为新能源车试验项，2024 全车型）──
    # 2021 与 2024 中文版正文 3.1.1.4 均为 32±0.5 km/h，75°±3°，刚性柱
    # 假人配置在两版间扩充（2026-08-15 中文版复核发现，T-4）：
    #   V6 2021：仅前排驾驶员位 WorldSID-50th
    #   V7 2024：驾驶员位 WorldSID-50th；**前排乘员位 ES-2re 或 WorldSID-50th**；
    #            **驾驶员后方座位 CRS + Q 系列 3 岁儿童假人**（用以测量第二排人员受伤害情况）
    "side_pole": {
        "speed_kmh": {"V6": 32, "V7": 32},
        "barrier": {"V6": "RIGID-POLE", "V7": "RIGID-POLE"},
        # row1 取驾驶员位假人。V7 前排乘员位可为 ES-2re 或 WorldSID-50th（规程允许二选一），
        # 本体 row1 不区分驾乘，此处以驾驶员位为准并在备注中声明（位置粒度见 T1 专题）
        "row1_dummy": {"V6": "WorldSID-50th", "V7": "WorldSID-50th"},
        # 修正：此前缺此项，导致 occ.side_pole.row2_child.* 的 σ.dummy 回退为 row1 的
        # WorldSID-50th（错误）。2024 中文版明载为 Q 系列 3 岁儿童假人。
        "row2_child_dummy": {"V7": "Q3"},
    },

    # ── 低速后碰撞颈部保护（鞭打）台架试验 ─────────────────────
    # 2012/2015 Δv=15.65；2018 起 Δv=20.0±1.0（2024 附录 E 确认 20.0±1.0）
    # 台架试验无壁障，显式标注而非留空
    "whiplash": {
        "delta_v_kmh": {"V3": 15.65, "V4": 15.65,
                        **_spread(20.0, ["V5", "V6", "V7"])},
        "barrier": _spread("SLED-NO-BARRIER", ["V3", "V4", "V5", "V6", "V7"]),
        "dummy": _spread("BioRID-II", ["V3", "V4", "V5", "V6", "V7"]),
    },

    # ── 侧面远端乘员保护（V7 新增，虚拟测评）───────────────────
    "side_farside": {
        "speed_kmh": {"V7": NA},
        "barrier": {"V7": "VIRTUAL"},
        "dummy": {"V7": "WorldSID-50th-VIRTUAL"},
    },

    # ── 非碰撞类评价项：显式标注无碰撞条件 ─────────────────────
    "child_static": {"speed_kmh": {**_spread(NA, ["V6", "V7"])},
                     "barrier": _spread(NA, ["V6", "V7"]),
                     "dummy": _spread(NA, ["V6", "V7"])},
    # 加分项（V3–V5）：由配置确定性给分，非碰撞试验，role = rule_unit_test
    "bonus": {"speed_kmh": _spread(NA, ["V3", "V4", "V5"]),
              "barrier": _spread(NA, ["V3", "V4", "V5"]),
              "dummy": _spread(NA, ["V3", "V4", "V5"])},
    "cpd": {"speed_kmh": {"V7": NA}, "barrier": {"V7": NA}, "dummy": {"V7": NA}},
    "ecall": {"speed_kmh": {"V7": NA}, "barrier": {"V7": NA}, "dummy": {"V7": NA}},
    "curtain_hold": {"speed_kmh": {"V7": NA}, "barrier": {"V7": NA}, "dummy": {"V7": NA}},
}

SIGMA_SEMANTICS = {
    "_doc": "σ =（碰撞速度, 壁障型号, 假人型号），用于判定同一 item_id 跨版本是否 comparable。",
    "_known_uncaptured": {
        "_doc": "σ 三维之外、已知但未纳入的测量条件差异。记录在此以免被当作「不存在」。",
        "side_mdb_impact_point": "撞击点位置由 R 点向后 250mm（V6）改为 200mm（V7）。"
                                 "该变更不改变可比性判定——V6→V7 的 side_mdb 已因速度（50→60）"
                                 "与壁障（AE-MDB→SC-MDB）判为 incomparable。故不扩展 σ 维度，仅登记。",
        "side_pole_passenger_dummy": "V7 前排乘员位可为 ES-2re 或 WorldSID-50th（规程允许二选一）。"
                                     "本体 row1 不区分驾乘，σ 以驾驶员位为准。位置粒度见 T1 专题。",
        "side_mdb_nonstruck_dummy": "V6 非撞击侧前排放置 ES-2 假人，规程明载「仅采集数据，暂不评价」，"
                                    "故不产生 item、不入 σ。V7 已取消该假人。",
    },
    "value_meaning": {
        "实际值": "该维度存在且已从规程原文抄录",
        "N/A-STATIC": "该工况存在但无此维度（静态或虚拟评价无碰撞速度／壁障）",
        "键不存在": "该工况在该版本不存在",
    },
    "comparability_rule": "同一 item_id 在版本 a 与 b 之间 comparable ⟺ 三个维度取值全同。任一维度变更即标 incomparable，不做数值折算。",
    "source": "全部参数出自规程原文，出处见 src/patch_sigma.py 的逐条注释。",
}


def dummy_key_for(scenario, dummy):
    """按 item 的 dummy 维度选取对应的假人字段。"""
    tbl = SIGMA.get(scenario, {})
    if "dummy" in tbl:                       # 单假人工况（鞭打、静态类）
        return "dummy"
    if dummy == "row2_child" and "row2_child_dummy" in tbl:
        return "row2_child_dummy"
    if dummy in ("row2_adult", "row2_child") and "row2_dummy" in tbl:
        return "row2_dummy"
    return "row1_dummy"


def sigma_for(scenario, dummy, version):
    tbl = SIGMA.get(scenario)
    if not tbl:
        return None
    dk = dummy_key_for(scenario, dummy)
    speed = tbl.get("speed_kmh", {}).get(version)
    dv = tbl.get("delta_v_kmh", {}).get(version)
    out = {}
    if speed is not None:
        out["speed_kmh"] = speed
    if dv is not None:
        out["delta_v_kmh"] = dv
    b = tbl.get("barrier", {}).get(version)
    if b is not None:
        out["barrier"] = b
    d = tbl.get(dk, {}).get(version)
    if d is not None:
        out["dummy"] = d
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    onto = json.loads(P.ONTO.read_text(encoding="utf-8"))
    before = json.dumps(onto.get("sigma_table"), ensure_ascii=False, sort_keys=True)

    onto["sigma_table"] = SIGMA
    onto["sigma_semantics"] = SIGMA_SEMANTICS

    stat = collections.Counter()
    for it in onto["items"]:
        scn, dmy = it["scenario"], it["dummy"]
        for v, blk in (it.get("versions") or {}).items():
            s = sigma_for(scn, dmy, v)
            old = blk.get("sigma")
            blk["sigma"] = s
            if s is None:
                stat["无σ"] += 1
            else:
                stat["已下发"] += 1
                for dim in ("speed_kmh", "delta_v_kmh", "barrier", "dummy"):
                    if dim in s:
                        stat[f"含{dim}"] += 1
                if not old or set(s) - set(old or {}):
                    stat["本次新增或补全"] += 1

    print("=== σ 下发统计 ===")
    for k, v in sorted(stat.items()):
        print(f"  {k}: {v}")
    changed = before != json.dumps(onto["sigma_table"], ensure_ascii=False, sort_keys=True)
    print(f"\nsigma_table {'已变更' if changed else '无变化'}")

    # 覆盖自检：每个 (scenario, version) 组合是否都有 barrier
    miss = []
    for it in onto["items"]:
        for v, blk in (it.get("versions") or {}).items():
            s = blk.get("sigma") or {}
            if "barrier" not in s:
                miss.append((it["scenario"], v))
    if miss:
        print("⚠ 仍缺 barrier 的 (工况, 版本)：", sorted(set(miss)))
    else:
        print("✔ 全部 (item, 版本) 均有 barrier")

    if a.dry_run:
        print("\n--dry-run，未写盘")
        return 0
    P.ONTO.write_text(json.dumps(onto, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n→ 已写入 {P.ONTO}")
    return 0


if __name__ == "__main__":
    P.require_raw("patch_sigma.py")
    sys.exit(main())
