# -*- coding: utf-8 -*-
"""
本地一键运行（Windows / macOS / Linux 通用）

用途：在装有 lightgbm 的本地环境重跑全部实验，并与沙箱内自实现 GBDT 的结果对比，
      作为「结论不依赖具体实现」的证据。

    python run_local.py              # 全链路：数据管线 → 校验 → 实验 → 对比
    python run_local.py --check      # 只做环境自检，不运行
    python run_local.py --skip-data  # 跳过数据管线（数据未变时省时间）

成功的标志：最后一节「实现一致性对比」中，两个后端的 skill 差异均在 ±0.05 以内。
"""
import argparse, io, os, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
OUT = HERE / "outputs"

# Windows 控制台默认代码页为 GBK(936)，中文输出会乱码。
# 仅设 PYTHONIOENCODING 不够——那只改 Python 侧编码，控制台仍按旧代码页解码。
# 必须同时把控制台代码页切到 UTF-8(65001)。
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_TTY = sys.stdout.isatty()
if _TTY and sys.platform == "win32":
    os.system("")          # 启用 ANSI 转义
C_OK, C_BAD, C_WARN, C_DIM, C_END = (
    ("\033[92m", "\033[91m", "\033[93m", "\033[90m", "\033[0m") if _TTY else ("",) * 5)


def say(msg, kind="info"):
    p = {"ok": (C_OK, "  ✔ "), "bad": (C_BAD, "  ✘ "), "warn": (C_WARN, "  ⚠ "),
         "info": ("", "    "), "step": ("", "")}[kind]
    print(f"{p[0]}{p[1]}{msg}{C_END}")


def head(t):
    print(f"\n{'='*74}\n{t}\n{'='*74}")


# ══════════════════ 环境自检 ══════════════════
def check_env():
    head("① 环境自检")
    ok = True

    v = sys.version_info
    if v >= (3, 8):
        say(f"Python {v.major}.{v.minor}.{v.micro}", "ok")
    else:
        say(f"Python {v.major}.{v.minor} 过旧，需 ≥3.8", "bad"); ok = False

    for mod, need in [("numpy", True), ("pandas", True), ("lightgbm", False),
                      ("sklearn", False), ("matplotlib", False)]:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            say(f"{mod} {ver}", "ok")
        except ImportError:
            if need:
                say(f"{mod} 缺失——必需。请先运行：pip install {mod}", "bad"); ok = False
            else:
                say(f"{mod} 未安装（可选）", "warn")

    try:
        import lightgbm  # noqa
        say("将使用 lightgbm 后端", "ok")
    except ImportError:
        say("未装 lightgbm，将回退到自实现 GBDT——本次运行无法提供实现对比证据", "warn")
        say("安装命令： pip install lightgbm", "info")

    # 路径与数据源
    sys.path.insert(0, str(SRC))
    try:
        import paths as P
        say(f"仓库根目录：{P.ROOT}", "ok")
        # Tier A 所需文件。原先这里检查的是原始抓取的列表页/详情页 CSV，
        # 但公开版不再分发原始数据，那两个常量为 None，导致 .exists() 抛
        # 'NoneType' object has no attribute 'exists'。
        need = [("细项长表", P.LONG_TABLE), ("车辆特征表", P.VEHICLE_FEAT),
                ("官方结果表", P.OFFICIAL), ("本体 JSON", P.ONTO),
                ("规则配置", P.RULE_CONFIG), ("特征分组", P.FEATURE_GROUPS)]
        miss = [n for n, q in need if q is None or not q.exists()]
        if miss:
            say("以下必需文件缺失：" + "、".join(miss), "bad"); ok = False
        else:
            say("Tier A 数据与配置齐全（无需原始抓取数据）", "ok")

        if P.RAW_AVAILABLE:
            say(f"检测到原始数据：{P.DATA}（Tier B 可用）", "ok")
        else:
            say("未设 CNCAP_RAW_DIR：build_long_table / validate_aggregation / "
                "patch_sigma 将跳过，其余全部可跑", "info")
    except Exception as e:
        say(f"路径模块加载失败：{e}", "bad"); ok = False

    return ok


# ══════════════════ 运行 ══════════════════
def run(script, args=(), timeout=1800):
    name = Path(script).name
    print(f"\n{C_DIM}$ python src/{name} {' '.join(args)}{C_END}")
    t0 = time.time()
    r = subprocess.run([sys.executable, str(SRC / name), *args],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, cwd=str(HERE))
    dt = time.time() - t0
    tail = [l for l in (r.stdout or "").splitlines() if l.strip()][-14:]
    for l in tail:
        print("   " + l)
    if r.returncode != 0:
        say(f"退出码 {r.returncode}（{dt:.0f}s）", "bad")
        for l in (r.stderr or "").splitlines()[-12:]:
            print("   " + l)
        return False, r.stdout or ""
    say(f"完成（{dt:.0f}s）", "ok")
    return True, r.stdout or ""


def detect_backend(text):
    m = re.search(r"GBDT 后端：(.+)", text)
    return m.group(1).strip() if m else "未知"


def snapshot(backend):
    """把本轮结果另存一份带后端标识的副本，供跨实现对比。"""
    tag = "lgb" if "lightgbm" in backend else ("skl" if "sklearn" in backend else "npy")
    # e5_min_config 也纳入：实测其边际贡献对后端敏感（正面气囊自检信号
    # 在自实现下恰为 0，在 lightgbm 下为 0.0026），故必须分后端留存。
    for f in ["baseline_group_cv.csv", "baseline_forward_transfer.csv",
              "ablation_e3.csv", "e5_min_config.csv"]:
        s = OUT / f
        if s.exists():
            shutil.copy(s, OUT / f"{s.stem}__{tag}{s.suffix}")
    return tag


def compare():
    """对比不同后端的前向迁移 skill。"""
    head("④ 实现一致性对比")
    try:
        import pandas as pd
    except ImportError:
        return
    files = {t: OUT / f"baseline_forward_transfer__{t}.csv" for t in ("npy", "lgb", "skl")}
    have = {t: p for t, p in files.items() if p.exists()}
    if len(have) < 2:
        say(f"当前只有 {len(have)} 个后端的结果，无法对比。", "warn")
        say("沙箱内已产出自实现结果（npy）。本地装 lightgbm 后重跑本脚本即可得到对比。", "info")
        return
    # 防护：若两份「不同后端」的文件内容完全相同，说明快照被覆盖，对比无效。
    # 曾发生过——不加这道检查会得出「差异 0.0000」的假一致性结论。
    import hashlib
    md5 = {t: hashlib.md5(p.read_bytes()).hexdigest() for t, p in have.items()}
    if len(set(md5.values())) < len(md5):
        say("检测到不同后端的结果文件内容完全相同——快照已被覆盖，对比无效。", "bad")
        say("处理：删除 outputs/baseline_forward_transfer__*.csv 后，先在无 lightgbm 的", "info")
        say("环境跑一次（得 __npy），再在有 lightgbm 的环境跑一次（得 __lgb）。", "info")
        return

    name = {"npy": "自实现", "lgb": "lightgbm", "skl": "sklearn"}
    dfs = {}
    for t, p in have.items():
        d = pd.read_csv(p, encoding="utf-8-sig")
        d = d[(d.model == "B2_gbdt") & (d.scope == "共有 item 子集")]
        dfs[t] = d.set_index("fold")["skill_MAE"]
    tab = pd.DataFrame({name[t]: s for t, s in dfs.items()})
    cols = list(tab.columns)
    tab["差异"] = (tab[cols].max(axis=1) - tab[cols].min(axis=1)).round(4)
    print("\nB2 在共有 item 子集上的 skill_MAE：\n")
    print(tab.round(4).to_string())
    mx = tab["差异"].max()
    print()
    # 判据分两层：数值一致性容易被小样本折破坏，结论方向才是关键
    pos = tab[tab.iloc[:, 0] > 0]
    mx_pos = pos["差异"].max() if len(pos) else 0.0
    signs_ok = all((tab.iloc[i, 0] > 0) == (tab.iloc[i, 1] > 0) for i in range(len(tab)))
    say(f"最大差异 {mx:.4f}（其中正 skill 折最大 {mx_pos:.4f}）", "info")
    if signs_ok:
        say("所有折的 skill 符号一致 —— 结论方向不依赖具体实现 ✔", "ok")
    else:
        say("存在符号不一致的折 —— 结论方向受实现影响，须谨慎", "bad")
    if mx_pos <= 0.05:
        say("正 skill 折的数值差异 ≤0.05，可直接引用", "ok")
    else:
        say("正 skill 折数值差异较大，报告时应给出区间而非点值", "warn")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只做环境自检")
    ap.add_argument("--skip-data", action="store_true", help="跳过数据管线")
    a = ap.parse_args()

    print(f"\n{'*'*74}\n  P01 本地运行　　工作目录：{HERE}\n{'*'*74}")

    if not check_env():
        print(f"\n{C_BAD}环境自检未通过，请按上方提示处理后重试。{C_END}")
        return 1
    if a.check:
        say("仅自检，未运行实验。去掉 --check 即可开始。", "info")
        return 0

    if not a.skip_data:
        head("② 数据管线与校验（约 1 分钟）")
        sys.path.insert(0, str(SRC))
        import paths as P
        # patch_sigma 与 build_long_table 属 Tier B：它们从原始记录重建数据集。
        # 公开版不分发原始记录，data/ 里已是重建后的产物，因此在没有
        # CNCAP_RAW_DIR 时应当跳过，而不是让整条流程中止。
        pipeline = [("patch_sigma.py", (), True), ("build_long_table.py", (), True),
                    ("validators.py", (), False), ("rule_layer.py", (), False),
                    ("sigma_analysis.py", (), False)]
        for s, args, needs_raw in pipeline:
            if needs_raw and not P.RAW_AVAILABLE:
                say(f"跳过 {s}（需原始数据，data/ 中已有其产物）", "info")
                continue
            ok, _ = run(s, args)
            if not ok:
                say(f"{s} 失败，已中止。上方错误信息即为原因。", "bad")
                return 1
    else:
        say("已跳过数据管线（--skip-data）", "warn")

    head("③ 实验（约 3–8 分钟，视机器而定）")
    ok, out1 = run("baselines.py", ("--cv", "both"))
    if not ok:
        return 1
    backend = detect_backend(out1)
    say(f"实际使用的后端：{backend}", "ok")

    # design_rules 必须先于 ablation_e3：后者读取前者产出的 C 类细项清单。
    run("design_rules.py")
    ok, _ = run("ablation_e3.py")
    if not ok:
        return 1
    run("monotonicity_check.py")
    # 表 7（oracle/zero 双模式 + 可观测性分层）与表 8（配置必要性）此前不在
    # 一键流程内，导致「单一入口复现全部结果」的说法不成立。已补入。
    run("oracle_zero.py")
    run("e5_min_config.py")

    tag = snapshot(backend)
    say(f"结果已另存为 *__{tag}.csv", "ok")

    compare()

    head("完成")
    print(f"""
  产出目录： {OUT}

  关键文件：
    baseline_group_cv.csv          同分布 5 折的分层结果
    baseline_forward_transfer.csv  前向迁移 5 折（2×2 析因主结果）
    ablation_e3.csv                单调约束消融 + C 类瓶颈细项
    design_rules_report.md         确定性设计准则清单
    oracle_zero.csv                oracle/zero 双模式（表 7）
    oracle_zero_star.csv           端到端星级
    e4_observability.csv           可观测性三分层
    e5_min_config.csv              配置必要性分析（表 8）

  下一步：把 outputs/ 目录下新增的 *__lgb.csv 发回对话，我来做交叉核对与结果判读。
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
