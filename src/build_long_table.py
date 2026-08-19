# -*- coding: utf-8 -*-
"""
长表数据集构建（W2/W3 交付物）

输出每行 = (车辆, 细项)，标签 y = 试验得分 / 满分 ∈ [0,1]。
结构性缺失表现为「不存在该行」，无需掩码机制。

用法：  python src/build_long_table.py
"""
import csv, json, glob, os, re, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import paths as P


# ────────────────────────────── 工具 ──────────────────────────────
def read_csv(path, enc="utf-8-sig"):
    with open(path, encoding=enc) as fh:
        return list(csv.DictReader(fh))


def fnum(x):
    """宽松数值解析，失败返回 None。"""
    if x is None:
        return None
    s = str(x).strip().replace(",", "").rstrip("%")
    if s in ("", "/", "-", "—", "N/A", "null", "None", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def record_id(fp):
    m = re.match(r"(\d+)_", os.path.basename(fp))
    return m.group(1) if m else None


# ────────────────────── 载入本体与映射 ──────────────────────
def load_ontology():
    onto = json.loads(P.ONTO.read_text(encoding="utf-8"))
    items = {i["item_id"]: i for i in onto["items"]}
    mapping, dropped = {}, {}
    for r in read_csv(P.MAPPING):
        key = (r["version"], r["raw_test"], r["raw_dummy"], r["raw_region"])
        if r["action"] == "map":
            mapping[key] = r["item_id"]
        else:
            dropped[key] = r["note"] or "已判定剔除"
    return onto, items, mapping, dropped


# ────────────────────── 车辆元数据与特征 ──────────────────────
def load_vehicle_meta():
    lst = {r["id"]: r for r in read_csv(P.LIST_CSV)}
    det = {r["detail_url"]: r for r in read_csv(P.DETAIL_CSV)}
    meta = {}
    for rid, r in lst.items():
        d = det.get(r["detail_url"])
        if not d:
            continue
        v = P.VERSION_MAP.get(d["regulation_version"])
        if not v:
            continue
        raw_overall = fnum(r.get("overall_score"))
        meta[rid] = {
            "variant_id": rid,
            # 内部连接键（下划线前缀者一律不落盘，保证产出物脱敏）
            "_car_name": (r.get("car_name") or "").strip(),
            "version": v,
            "test_year": r.get("test_year"),
            "vehicle_class_raw": r.get("car_kind"),
            "star_official": fnum(d.get("star_rating")),
            # D-019：该列语义随版本变化，此处显式拆分
            "total_score": raw_overall if v in ("V1", "V2", "V3", "V4") else None,
            "overall_rate": (raw_overall / 100 if raw_overall and raw_overall > 1 else raw_overall)
                            if v in ("V5", "V6", "V7") else None,
            "occupant_rate": (lambda x: x / 100 if x and x > 1 else x)(fnum(r.get("ap_score"))),
            "vru_rate":      (lambda x: x / 100 if x and x > 1 else x)(fnum(r.get("vru_score"))),
            "active_rate":   (lambda x: x / 100 if x and x > 1 else x)(fnum(r.get("as_score"))),
        }
    return meta


YES = lambda s: 1 if s and str(s).strip() not in ("", "无", "-", "—", "否", "null", "N/A") else 0

# 特征抽取过程的质检计数（并入 qc_report.json）
FEATURE_QC = {}


# ── D-048：早期版本 safety_configs 部件级文本块的规则化抽取 ──────────
# 备注为半结构化短语，关键词匹配即可，不使用 LLM。所抽取变量均与乘员保护
# 有明确因果通路（安全带限力/预张紧、气帘、膝部气囊、头枕类型）。
def _blank(s):
    return str(s or "").strip() in ("", "无", "-", "—", "―", "――", "--", "/", "null", "N/A")


def _belt_grade(remark):
    """安全带三档：2=预张紧+限力 / 1=仅限力 / 0=普通三点式 / None=未记录

    官方备注存在肯定式（「预张紧及限力功能」）与枚举式（「预张紧无、限力功能有」）
    两种写法，须分别判定，否则枚举式的否定会被误读为肯定。
    """
    t = re.sub(r"\s+", "", str(remark or ""))
    if not t or _blank(t):
        return None
    if "普通三点式" in t:
        return 0
    # 枚举式：显式声明有/无
    pre_neg = ("预张紧无" in t) or ("无预张紧" in t)
    lim_neg = ("限力功能无" in t) or ("无限力" in t)
    has_pre = ("预张紧" in t or "预紧" in t) and not pre_neg
    has_lim = ("限力" in t) and not lim_neg
    if has_pre and has_lim:
        return 2
    if has_lim or has_pre:
        return 1
    if pre_neg and lim_neg:
        return 0        # 明确声明两者皆无 == 普通三点式，不是「未记录」
    return None


# 供应商原始字段极脏：全角/半角括号、城市前缀或后缀、全称与简称并存、
# 驾乘两侧同名拼接（如「锦州锦恒\n \n锦州锦恒」）。基数 249/330 近乎每台一值，
# 直接作类别特征必然过拟合。故规范化到**品牌级**，未识别者归 other。
SUPPLIER_BRANDS = [
    ("autoliv", ("奥托立夫", "AUTOLIV")), ("takata", ("高田", "TAKATA")),
    ("joyson", ("延锋百利得", "百利得", "均胜", "JOYSON")),
    ("jinheng", ("锦恒",)), ("trw_zf", ("天合", "采埃孚", "TRW", "ZF")),
    ("dongfang_jiule", ("东方久乐",)), ("guangda", ("光大",)),
    ("huamao", ("华懋",)), ("shuanglin", ("双林",)),
    ("adient_yfjc", ("延锋江森", "安道拓", "江森", "ADIENT")),
    ("lear", ("李尔", "LEAR")), ("faurecia", ("佛吉亚", "FAURECIA")),
    ("tachis", ("提爱思", "TACHI")), ("fuwei", ("富维",)),
    ("honglizhixin", ("宏立至信",)), ("toyota_boshoku", ("丰田纺织", "丰田自动织机")),
    ("magna", ("麦格纳", "MAGNA")), ("brose", ("博泽", "BROSE")),
    ("keiper_recaro", ("凯波", "recaro", "RECARO")),
]


def normalize_supplier(s):
    """原始供应商文本 → 品牌级标识；无法识别返回 'other'，空返回 None。"""
    if _blank(s):
        return None
    t = re.sub(r"\s+", "", str(s)).replace("（", "(").replace("）", ")").upper()
    for brand, keys in SUPPLIER_BRANDS:
        if any(k.upper() in t for k in keys):
            return brand
    return "other"


def extract_component_features(f, items):
    for it in items:
        sysname = str(it.get("safety_system") or "").strip()
        remark = str(it.get("remark") or "")
        t = re.sub(r"\s+", "", remark)
        mfr = None if _blank(it.get("manufacturer")) else str(it["manufacturer"]).strip()
        mdl = None if _blank(it.get("model")) else str(it["model"]).strip()

        if "驾驶员座椅" in sysname:
            hr = None
            if "主动式头枕" in t and "非主动式" not in t:
                hr = "active"
            elif "非主动式头枕" in t:
                hr = "passive"
            f["headrest_type"] = hr
            f["seat_supplier"], f["seat_model"] = normalize_supplier(mfr), mdl

        elif "前排安全带" in sysname:
            f["belt_grade_row1"] = _belt_grade(remark)
            f["belt_supplier"] = normalize_supplier(mfr)

        elif "第二排" in sysname and "安全带" in sysname:
            f["belt_grade_row2"] = _belt_grade(remark)

        elif "正面安全气囊" in sysname:
            f["airbag_supplier"] = normalize_supplier(mfr)
            if not _blank(remark):
                f["cfg_knee_airbag"] = 1 if "膝部" in t else 0

        elif "侧面安全气囊" in sysname:
            if not _blank(remark):
                f["cfg_side_airbag"] = 1 if "侧气囊" in t or "侧面气囊" in t else 0
                f["cfg_curtain_airbag"] = 1 if "气帘" in t else 0
            f["side_airbag_supplier"] = normalize_supplier(mfr)


def load_features(meta):
    """A 级整车参数 + B 级功能级配置二值特征 + 头枕字段（D-024）。"""
    feat = {rid: {"variant_id": rid} for rid in meta}

    # —— 早期版本：供应商级自由文本 ——
    for fp in glob.glob(str(P.CFG_EARLY_DIR / "*.json")):
        rid = record_id(fp)
        if rid not in feat:
            continue
        d = json.loads(Path(fp).read_text(encoding="utf-8"))
        bi = d.get("basic_info") or {}
        f = feat[rid]
        f.update({
            "length_mm": fnum(bi.get("length_mm")), "width_mm": fnum(bi.get("width_mm")),
            "height_mm": fnum(bi.get("height_mm")), "curb_weight_kg": fnum(bi.get("curb_weight_kg")),
            "gross_weight_kg": fnum(bi.get("gross_vehicle_weight_kg")),
            "displacement_ml": fnum(bi.get("engine_displacement_ml")),
            "price_10k": fnum(bi.get("guide_price_of_test_model_ten_thousand_yuan")),
            "cfg_front_airbag": YES(bi.get("front_airbags")),
            "cfg_side_airbag": YES(bi.get("side_airbags")),
            "cfg_curtain_airbag": YES(bi.get("curtain_airbags")),
            "cfg_belt_pretensioner": YES(bi.get("seatbelt_pre_tensioners")),
            "cfg_belt_loadlimiter": YES(bi.get("seatbelt_force_limiters")),
            "cfg_belt_reminder_driver": YES(bi.get("driver_seatbelt_reminder")),
            "cfg_belt_reminder_passenger": YES(bi.get("occupant_seatbelt_reminder")),
            "cfg_isofix": YES(bi.get("isofix")),
            "cfg_esc": YES(bi.get("esc")),
            "cfg_seat_monitor": YES(bi.get("occupant_seat_monitoring")),
            "feature_source": "early_supplier_text",
        })
        # D-024 / D-048：开采 safety_configs 部件级文本块（8 条目 × 供应商/型号/备注）
        extract_component_features(f, (d.get("safety_configs") or {}).get("items") or [])

    # —— 近期版本 A 级整车参数：独立 CSV，按车名 1:1 连接（W4 修复，见 D-044）——
    name2rid = {m0["_car_name"]: rid for rid, m0 in meta.items() if m0["_car_name"]}
    late_basic_hit = 0
    if P.CFG_LATE_BASIC.exists():
        for r in read_csv(P.CFG_LATE_BASIC):
            rid = name2rid.get((r.get("car_name") or "").strip())
            if rid is None:
                continue
            late_basic_hit += 1
            feat[rid].update({
                "length_mm": fnum(r.get("length_mm")), "width_mm": fnum(r.get("width_mm")),
                "height_mm": fnum(r.get("height_mm")),
                "curb_weight_kg": fnum(r.get("curb_weight_kg")),
                "displacement_ml": fnum(r.get("engine_displacement_ml")),
                "price_10k": fnum(r.get("guide_price_of_test_model_ten_thousand_yuan")),
            })

    # —— 近期版本：座位粒度矩阵 ——
    # W4 发现（F-31）：「其他安全配置」键名存在全角/半角括号与空格三重排版变体，
    # 同一语义最多四种写法。此处以英文缩写为准归一，中文项单列。
    def norm_key(k):
        return re.sub(r"\s+", "", str(k).replace("（", "(").replace("）", ")"))

    ABBR2COL = {"ESC": "cfg_esc", "LKA": "cfg_lka", "LDW": "cfg_ldw", "BSD": "cfg_bsd",
                "SAS": "cfg_sas", "ADB": "cfg_adb", "E-CALL": "cfg_ecall",
                "ELK": "cfg_elk", "DMS": "cfg_dms", "TSR": "cfg_tsr",
                "ISLS": "cfg_isls", "DOW": "cfg_dow", "RCTA": "cfg_rcta"}
    AEB_SUFFIX = {"车-车": "cfg_aeb_c2c", "车-行人": "cfg_aeb_ped", "车-二轮车": "cfg_aeb_tw"}

    def active_col(k):
        """归一化键 → 特征列名；无法识别返回 None。"""
        k = norm_key(k)
        if k.startswith("AEB"):
            tail = k.split(")")[-1]
            return AEB_SUFFIX.get(tail)
        if k == "主动弹起式发动机罩":
            return "cfg_active_bonnet"
        abbr = re.match(r"^([A-Z][A-Z\-]*)", k)
        return ABBR2COL.get(abbr.group(1)) if abbr else None

    late_ok, late_fail = 0, []
    for fp in glob.glob(str(P.CFG_LATE_DIR / "*.json")):
        rid = record_id(fp)
        if rid not in feat:
            continue
        d = json.loads(Path(fp).read_text(encoding="utf-8"))
        txt = d.get("extracted_data") or ""
        m = re.search(r"\{.*\}", txt, re.S)
        sc = None
        if m:
            try:
                sc = json.loads(m.group(0)).get("safety_configuration") or {}
            except Exception:
                sc = None
        if sc is None:
            late_fail.append(rid)          # 源文件即抽取失败（文件名含 _error_）
            continue
        late_ok += 1
        f = feat[rid]
        f["feature_source"] = "late_seat_matrix"
        for k, v in (sc.get("其他安全配置") or {}).items():
            col = active_col(k)
            if col:
                f[col] = 1 if str(v).strip() == "√" else 0
        # 座位粒度 → 功能级折叠（B 级最小公共集）
        def _has_check(o):
            """递归查找 √。2021 版起部分项为双层嵌套（如「侧面安全气囊」下再分
            「胸部保护」「头胸一体式」，各自再分四个座位），只看一层会恒为 0。"""
            if isinstance(o, dict):
                return any(_has_check(x) for x in o.values())
            return str(o).strip() == "√"

        def any_pos(block, key):
            return 1 if _has_check((sc.get(block) or {}).get(key) or {}) else 0
        f["cfg_front_airbag"] = any_pos("正面碰撞保护", "正面安全气囊")
        f["cfg_knee_airbag"] = any_pos("正面碰撞保护", "膝部安全气囊")
        f["cfg_side_airbag"] = any_pos("侧面碰撞保护", "侧面安全气囊")
        f["cfg_curtain_airbag"] = any_pos("侧面碰撞保护", "侧面安全气帘")
        f["cfg_center_airbag"] = any_pos("侧面碰撞保护", "中央安全气囊")
        f["cfg_isofix"] = any_pos("座椅安全带配置", "ISOFIX装置")
        f["cfg_belt_pretensioner"] = any_pos("座椅安全带配置", "安全带预张紧器")
        f["cfg_belt_loadlimiter"] = any_pos("座椅安全带配置", "安全带限力器")
        f["cfg_seat_monitor"] = any_pos("座椅安全带配置", "座椅使用状态监测")
        f["cfg_belt_reminder_driver"] = any_pos("座椅安全带配置", "安全带未系提醒")

    FEATURE_QC.update({"late_basic_joined": late_basic_hit,
                       "late_matrix_parsed": late_ok,
                       "late_matrix_failed": sorted(late_fail)})

    # 尺寸等基础参数回填自列表页（早期文件缺失时）
    for rid, m0 in meta.items():
        feat[rid].setdefault("feature_source", "meta_only")
        feat[rid]["version"] = m0["version"]
        feat[rid]["test_year"] = m0["test_year"]
        feat[rid]["vehicle_class_raw"] = m0["vehicle_class_raw"]
    return feat


# ────────────────────── 细项抽取 ──────────────────────
def iter_test_level(occ):
    """产出 (raw_test, score, denom)：试验项自身的官方得分。"""
    blocks = list(occ.get("乘员保护试验列表") or [])
    for k, v in occ.items():
        if isinstance(v, dict) and "试验得分" in v:
            blocks.append({**v, "试验项": v.get("试验项") or k})
    for t in blocks:
        tn, s = t.get("试验项"), str(t.get("试验得分") or "")
        if tn and "/" in s:
            try:
                num, den = s.split("/")
                yield tn, float(num), float(den)
            except ValueError:
                continue


def iter_raw_items(occ):
    """产出 (raw_test, raw_dummy, raw_region, score, max)。兼容两类容器。"""
    for t in (occ.get("乘员保护试验列表") or []):
        tn = t.get("试验项")
        if tn is None:
            continue
        if isinstance(t.get("子项"), list):          # 容器 A：子项直挂
            for s in t["子项"]:
                if isinstance(s, dict) and "试验名称" in s:
                    yield tn, "子项", s["试验名称"], fnum(s.get("试验得分")), fnum(s.get("满分"))
        for dm, blk in t.items():                    # 容器 B：按假人组分块
            if not isinstance(blk, dict):
                continue
            for s in (blk.get("子项") or blk.get(dm) or []):
                if isinstance(s, dict) and "试验名称" in s:
                    yield tn, dm, s["试验名称"], fnum(s.get("试验得分")), fnum(s.get("满分"))
    for k, v in occ.items():                         # 容器 C：模块下的兄弟块（鞭打/加分项）
        if k in ("乘员保护试验列表", "乘员保护得分率", "乘员保护试验得分", "试验项") or not isinstance(v, dict):
            continue
        for s in (v.get("子项") or []):
            if isinstance(s, dict):
                nm = s.get("试验名称") or s.get("项目名称")
                if nm:
                    yield (v.get("试验项") or k), "子项", nm, fnum(s.get("试验得分")), fnum(s.get("满分"))


def main():
    onto, items, mapping, dropmap = load_ontology()
    test2scn = {}
    for (vv, rt0, _, _), iid0 in mapping.items():
        if iid0.startswith("occ."):
            test2scn[(vv, rt0)] = iid0.split(".")[1]
    meta = load_vehicle_meta()
    feats = load_features(meta)
    corr = {(c["target"], c["version"]): c["to"]
            for c in onto.get("corrections", []) if "version" in c}

    rows, unmapped, stats = [], collections.Counter(), collections.Counter()
    outliers = collections.Counter()
    official_tl = {}
    seen = set()
    for pattern in P.MODULE_GLOBS:
        for fp in glob.glob(str(pattern)):
            rid = record_id(fp)
            m0 = meta.get(rid)
            if not m0:
                stats["no_meta"] += 1
                continue
            v = m0["version"]
            if (v, rid) in P.BAD_RECORDS:
                stats["bad_record_skipped"] += 1
                continue
            occ = (json.loads(Path(fp).read_text(encoding="utf-8")) or {}).get("乘员保护模块") or {}

            # —— 试验项级官方得分：既用于 TEST_LEVEL 工况建项，也用于计算调整量 ——
            tl = {}
            for rt, sc0, den0 in iter_test_level(occ):
                scn0 = test2scn.get((v, rt))
                if scn0:
                    tl[scn0] = (sc0, den0)
            for scn0, (sc0, den0) in tl.items():
                if scn0 not in P.TEST_LEVEL_SCENARIOS:
                    continue
                iid = f"occ.{scn0}.total"
                if (rid, iid) in seen:
                    continue
                seen.add((rid, iid))
                rows.append({
                    "variant_id": rid, "version": v, "test_year": m0["test_year"],
                    "item_id": iid, "item_id_coarse": iid,
                    "scenario": scn0, "dummy": "", "region": "total",
                    "role": "target", "observability_class": "config_gated",
                    "score": sc0, "max": den0,
                    "y": sc0 / den0 if den0 else None,
                    "sigma_speed": None, "sigma_barrier": None, "sigma_dummy": None,
                })
                stats["test_level_items"] += 1
            official_tl[(rid, v)] = tl

            for rt, rd, rr, score, mx in iter_raw_items(occ):
                iid = mapping.get((v, rt, rd, rr))
                if iid is None:
                    k = (v, rt, rd, rr)
                    if k in dropmap:
                        stats["intentionally_dropped"] += 1
                    else:
                        unmapped[k] += 1
                        stats["unmapped"] += 1
                    continue
                key = (rid, iid)
                if key in seen:                       # G2/G3：容器重复，去重
                    stats["dedup"] += 1
                    continue
                seen.add(key)
                it = items.get(iid)
                if it is None:
                    continue
                if it["scenario"] in P.TEST_LEVEL_SCENARIOS:
                    stats["test_level_subitems_skipped"] += 1
                    seen.discard(key)
                    continue
                mx = corr.get((iid, v), mx)           # 应用已知修正
                vinfo = it["versions"].get(v, {})
                exp = corr.get((iid, v), vinfo.get("max"))
                if exp is not None and mx is not None and abs(mx - exp) > 1e-6:
                    stats["max_outlier_dropped"] += 1
                    outliers[(v, iid, mx, exp)] += 1
                    seen.discard(key)
                    continue
                sigma = vinfo.get("sigma") or {}
                y = (score / mx) if (score is not None and mx not in (None, 0)) else None
                rows.append({
                    "variant_id": rid, "version": v, "test_year": m0["test_year"],
                    "item_id": iid, "item_id_coarse": it["item_id_coarse"],
                    "scenario": it["scenario"], "dummy": it["dummy"], "region": it["region"],
                    "role": it["role"], "observability_class": it["observability_class"],
                    "score": score, "max": mx, "y": y,
                    "sigma_speed": sigma.get("speed_kmh") or sigma.get("delta_v_kmh"),
                    "sigma_barrier": sigma.get("barrier"),
                    "sigma_dummy": sigma.get("dummy"),
                })
                stats["rows"] += 1

    # ── 并入补抓数据（V6 侧碰第二排）──
    sup = json.loads(P.SUPPLEMENT.read_text(encoding="utf-8"))
    n_sup = 0
    for rec in sup["supplements"]:
        vid = rec["variant_id"]
        m0 = meta.get(vid)
        if not m0:
            continue
        for it0 in rec["items"]:
            iid = f"occ.side_mdb.row2_adult.{it0['region']}"
            if (vid, iid) in seen:
                continue
            seen.add((vid, iid))
            it = items.get(iid)
            if it is None:
                continue
            sigma = (it["versions"].get(m0["version"], {}) or {}).get("sigma") or {}
            rows.append({
                "variant_id": vid, "version": m0["version"], "test_year": m0["test_year"],
                "item_id": iid, "item_id_coarse": it["item_id_coarse"],
                "scenario": it["scenario"], "dummy": it["dummy"], "region": it["region"],
                "role": it["role"], "observability_class": it["observability_class"],
                "score": it0["score"], "max": it0["max"],
                "y": it0["score"] / it0["max"] if it0["max"] else None,
                "sigma_speed": sigma.get("speed_kmh") or sigma.get("delta_v_kmh"),
                "sigma_barrier": sigma.get("barrier"), "sigma_dummy": sigma.get("dummy"),
            })
            n_sup += 1
    stats["supplemented"] = n_sup

    # 落盘
    import gzip
    cols = list(rows[0].keys())
    with gzip.open(P.LONG_TABLE, "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    fcols = sorted({k for f in feats.values() for k in f})
    fcols = ["variant_id", "version"] + [c for c in fcols if c not in ("variant_id", "version")]
    with open(P.VEHICLE_FEAT, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fcols)
        w.writeheader()
        for rid in sorted(feats, key=lambda x: int(x)):
            w.writerow(feats[rid])

    # ── 试验项级调整表：官方试验得分 − Σ细项得分 ──
    isum = collections.defaultdict(float)
    for r in rows:
        if r["role"] in ("target", "aggregation_source") and r["score"] is not None:
            if r["scenario"] not in P.TEST_LEVEL_SCENARIOS:
                isum[(r["variant_id"], r["scenario"])] += r["score"]
    adj_rows = []
    for (rid, v0), tl in official_tl.items():
        for scn0, (sc0, den0) in tl.items():
            if scn0 in P.TEST_LEVEL_SCENARIOS:
                continue
            key = (rid, scn0)
            if key not in isum:
                continue
            a = round(sc0 - isum[key], 4)
            adj_rows.append({"variant_id": rid, "version": v0, "scenario": scn0,
                             "item_sum": round(isum[key], 4), "official_test_score": sc0,
                             "denominator": den0, "adjustment": a,
                             "kind": ("none" if abs(a) < 1e-3 else
                                      "override_zero" if abs(sc0) < 1e-9 and isum[key] > 1e-3 else
                                      "penalty" if a < 0 else "bonus")})
    with open(P.ADJUSTMENTS, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(adj_rows[0].keys()))
        w.writeheader(); w.writerows(adj_rows)
    adj_kind = collections.Counter(a["kind"] for a in adj_rows)

    # 汇总
    by_v = collections.Counter(r["version"] for r in rows)
    by_role = collections.Counter(r["role"] for r in rows)
    bad_y = sum(1 for r in rows if r["role"] == "target" and r["y"] is not None and not (0 <= r["y"] <= 1))
    hr = collections.Counter(f.get("headrest_type") for f in feats.values())
    report = {
        "generated": "auto",
        "n_rows": len(rows), "n_vehicles": len(set(r["variant_id"] for r in rows)),
        "rows_by_version": dict(by_v), "rows_by_role": dict(by_role),
        "dedup_removed": stats["dedup"], "intentionally_dropped": stats["intentionally_dropped"],
        "unmapped_rows": stats["unmapped"],
        "bad_records_skipped": stats["bad_record_skipped"],
        "max_outlier_dropped": stats["max_outlier_dropped"],
        "supplemented_rows": stats["supplemented"],
        "max_outliers": [{"key": list(k), "n": n} for k, n in outliers.most_common(10)],
        "label_out_of_range": bad_y,
        "test_level_items": stats["test_level_items"],
        "test_level_subitems_skipped": stats["test_level_subitems_skipped"],
        "adjustment_kinds": dict(adj_kind),
        "headrest_coverage": {str(k): v for k, v in hr.items()},
        "feature_extraction": dict(FEATURE_QC),
        "feature_coverage": {c: round(sum(1 for f in feats.values()
                                          if f.get(c) not in (None, "")) / len(feats), 4)
                             for c in sorted({k for f in feats.values() for k in f})
                             if c not in ("variant_id", "version")},
        "unmapped_top": [{"key": list(k), "n": n} for k, n in unmapped.most_common(15)],
    }
    P.QC_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"长表行数 {len(rows)}｜车辆 {report['n_vehicles']}｜去重 {stats['dedup']}｜按判定剔除 {stats['intentionally_dropped']}｜未映射(异常) {stats['unmapped']}")
    print("按版本:", dict(sorted(by_v.items())))
    print("按角色:", dict(by_role))
    print("标签越界:", bad_y)
    print("补抓并入:", stats["supplemented"], "行")
    print("试验项级 item:", stats["test_level_items"], "｜跳过子项:", stats["test_level_subitems_skipped"])
    print("试验项级调整:", dict(adj_kind))
    print("头枕字段:", dict(hr))
    if outliers:
        print("满分离群剔除:", stats["max_outlier_dropped"], "→", [ (k[0],k[1].split(".")[-1],f"{k[2]}≠{k[3]}",n) for k,n in outliers.most_common(4) ])
    if unmapped:
        print("未映射样例:")
        for k, n in unmapped.most_common(5):
            print("   ", k, n)


if __name__ == "__main__":
    P.require_raw("build_long_table.py")
    main()
