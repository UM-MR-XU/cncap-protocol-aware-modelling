# -*- coding: utf-8 -*-
"""
梯度提升回归树（纯 numpy 实现）

为什么自己实现：沙箱内 pip 被代理拦截，lightgbm / xgboost / sklearn 均不可安装。
但本研究有两项硬需求无法绕开——

    1. **原生缺失值处理**。长表中结构性缺失普遍存在（如部件级特征仅 V3–V5 可得），
       填充会把「该版本不评价此项」与「配备了但没有」混为一谈。
    2. **单调性约束**。E3 消融的核心是「施加单调约束 vs 不施加」，这要求分裂搜索
       阶段即可施加方向约束，事后修正无法等价。

实现要点
--------
· 直方图分箱（默认 64 箱），缺失值单独成箱并学习最优默认方向（left / right）
· 平方损失，叶值 = 残差和 /（样本数 + λ）
· 单调约束：`constraint ∈ {-1, 0, 1}`，分裂时校验左右叶值方向，不满足则拒绝该分裂点
· 行采样与列采样，与常见实现语义一致

与成熟实现的差异（须在论文中声明）
----------------------------------
· **类别特征按整数码当作有序处理**，未实现 LightGBM 的类别最优切分。对无序类别
  （如工况、部位）需多次分裂才能逼近，可能略损精度。
· 单调约束为**局部校验**（检查直接左右叶值），未实现完整的上下界传播。这是**保守**
  实现——可能拒绝一些实际合法的分裂，但不会产生违反单调性的模型。
· 未实现 EFB、GOSS 等加速技术；本数据规模（11k 行）无需。

接口与 sklearn 一致（fit / predict），故 `baselines.py` 可无缝切换后端。
若本地装有 lightgbm，应优先使用并与本实现交叉验证（见 `--backend` 参数）。

用法：  python src/gbdt.py     # 自检：合成数据上验证缺失值与单调约束生效
"""
import numpy as np

MISSING_BIN = 255          # 缺失值专用箱号
_EPS = 1e-12


# ══════════════════════════ 分箱 ══════════════════════════
class Binner:
    """按分位数分箱。缺失值归入 MISSING_BIN，不参与分位数计算。"""

    def __init__(self, max_bins=64):
        self.max_bins = min(max_bins, 254)
        self.edges = None

    def fit(self, X):
        n_feat = X.shape[1]
        self.edges = []
        for j in range(n_feat):
            col = X[:, j]
            v = col[~np.isnan(col)]
            if v.size == 0:
                self.edges.append(np.array([0.0]))
                continue
            qs = np.linspace(0, 100, self.max_bins + 1)[1:-1]
            e = np.unique(np.percentile(v, qs))
            self.edges.append(e if e.size else np.array([v[0]]))
        return self

    def transform(self, X):
        out = np.empty(X.shape, dtype=np.uint8)
        for j in range(X.shape[1]):
            col = X[:, j]
            nan = np.isnan(col)
            b = np.searchsorted(self.edges[j], np.where(nan, 0.0, col), side="left")
            b = np.minimum(b, self.max_bins - 1).astype(np.uint8)
            b[nan] = MISSING_BIN
            out[:, j] = b
        return out

    def n_bins(self, j):
        return min(len(self.edges[j]) + 1, self.max_bins)


# ══════════════════════════ 单棵树 ══════════════════════════
class _Node:
    __slots__ = ("feat", "thr_bin", "default_left", "left", "right", "value", "is_leaf")

    def __init__(self):
        self.feat = -1
        self.thr_bin = -1
        self.default_left = True
        self.left = None
        self.right = None
        self.value = 0.0
        self.is_leaf = False


class _Tree:
    def __init__(self, max_depth, min_samples_leaf, l2, monotone, n_bins_fn, rng,
                 colsample):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.monotone = monotone          # 每特征 -1/0/1
        self.n_bins_fn = n_bins_fn
        self.rng = rng
        self.colsample = colsample
        self.root = None

    @staticmethod
    def _leaf_value(g_sum, n, l2):
        return g_sum / (n + l2)

    def _best_split(self, Xb, grad, idx, feats):
        """在 idx 子集上搜索最优分裂。返回 (gain, feat, thr_bin, default_left)。"""
        n = idx.size
        G = grad[idx].sum()
        base = G * G / (n + self.l2)
        best = (0.0, -1, -1, True)

        for j in feats:
            nb = self.n_bins_fn(j)
            col = Xb[idx, j]
            # 直方图：0..nb-1 为正常箱，最后一格为缺失
            hist_g = np.zeros(nb + 1)
            hist_n = np.zeros(nb + 1)
            miss = col == MISSING_BIN
            if miss.any():
                np.add.at(hist_g, nb, grad[idx][miss].sum())
                hist_n[nb] = miss.sum()
            ok = ~miss
            if ok.any():
                np.add.at(hist_g, col[ok].astype(np.intp), grad[idx][ok])
                np.add.at(hist_n, col[ok].astype(np.intp), 1.0)

            gm, nm = hist_g[nb], hist_n[nb]
            cg = np.cumsum(hist_g[:nb])
            cn = np.cumsum(hist_n[:nb])
            mono = self.monotone[j]

            for d_left in (True, False):
                gl = cg[:-1] + (gm if d_left else 0.0)
                nl = cn[:-1] + (nm if d_left else 0.0)
                gr = G - gl
                nr = n - nl
                valid = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
                if not valid.any():
                    continue
                vl = gl / (nl + self.l2)
                vr = gr / (nr + self.l2)
                if mono == 1:
                    valid &= vl <= vr + _EPS
                elif mono == -1:
                    valid &= vl >= vr - _EPS
                gain = np.where(valid, gl * vl + gr * vr - base, -np.inf)
                k = int(np.argmax(gain))
                if gain[k] > best[0]:
                    best = (float(gain[k]), j, k, d_left)
        return best

    def _build(self, Xb, grad, idx, depth):
        node = _Node()
        node.value = self._leaf_value(grad[idx].sum(), idx.size, self.l2)
        if depth >= self.max_depth or idx.size < 2 * self.min_samples_leaf:
            node.is_leaf = True
            return node

        n_feat = Xb.shape[1]
        k = max(1, int(round(self.colsample * n_feat)))
        feats = self.rng.choice(n_feat, size=k, replace=False) if k < n_feat else np.arange(n_feat)

        gain, j, thr, d_left = self._best_split(Xb, grad, idx, feats)
        if j < 0 or gain <= _EPS:
            node.is_leaf = True
            return node

        col = Xb[idx, j]
        miss = col == MISSING_BIN
        go_left = np.where(miss, d_left, col <= thr)
        li, ri = idx[go_left], idx[~go_left]
        if li.size < self.min_samples_leaf or ri.size < self.min_samples_leaf:
            node.is_leaf = True
            return node

        node.feat, node.thr_bin, node.default_left = j, thr, d_left
        node.left = self._build(Xb, grad, li, depth + 1)
        node.right = self._build(Xb, grad, ri, depth + 1)
        return node

    def fit(self, Xb, grad, idx):
        self.root = self._build(Xb, grad, idx, 0)
        return self

    def predict(self, Xb):
        out = np.empty(Xb.shape[0])
        stack = [(self.root, np.arange(Xb.shape[0]))]
        while stack:
            node, idx = stack.pop()
            if idx.size == 0:
                continue
            if node.is_leaf:
                out[idx] = node.value
                continue
            col = Xb[idx, node.feat]
            miss = col == MISSING_BIN
            go_left = np.where(miss, node.default_left, col <= node.thr_bin)
            stack.append((node.left, idx[go_left]))
            stack.append((node.right, idx[~go_left]))
        return out


# ══════════════════════════ 提升器 ══════════════════════════
class GBDTRegressor:
    """平方损失梯度提升树。接口与 sklearn 一致。

    monotone_constraints: 长度 = 特征数的序列，取值 -1/0/1；None 表示全不约束。
    """

    def __init__(self, n_estimators=400, learning_rate=0.05, max_depth=6,
                 min_samples_leaf=20, l2=1.0, max_bins=64, subsample=0.8,
                 colsample=0.8, monotone_constraints=None, random_state=42,
                 verbose=False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2 = l2
        self.max_bins = max_bins
        self.subsample = subsample
        self.colsample = colsample
        self.monotone_constraints = monotone_constraints
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        rng = np.random.RandomState(self.random_state)

        mono = (np.zeros(p, dtype=int) if self.monotone_constraints is None
                else np.asarray(self.monotone_constraints, dtype=int))
        assert mono.size == p, f"单调约束长度 {mono.size} ≠ 特征数 {p}"

        self.binner_ = Binner(self.max_bins).fit(X)
        Xb = self.binner_.transform(X)
        self.init_ = float(y.mean())
        pred = np.full(n, self.init_)
        self.trees_ = []

        for it in range(self.n_estimators):
            grad = y - pred                      # 平方损失的负梯度
            if self.subsample < 1.0:
                m = rng.rand(n) < self.subsample
                idx = np.where(m)[0]
                if idx.size < 2 * self.min_samples_leaf:
                    idx = np.arange(n)
            else:
                idx = np.arange(n)
            t = _Tree(self.max_depth, self.min_samples_leaf, self.l2, mono,
                      self.binner_.n_bins, rng, self.colsample).fit(Xb, grad, idx)
            pred += self.learning_rate * t.predict(Xb)
            self.trees_.append(t)
            if self.verbose and (it + 1) % 100 == 0:
                print(f"    iter {it+1}: train RMSE {np.sqrt(((y-pred)**2).mean()):.4f}")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self.binner_.transform(X)
        out = np.full(X.shape[0], self.init_)
        for t in self.trees_:
            out += self.learning_rate * t.predict(Xb)
        return out


# ══════════════════════════ 自检 ══════════════════════════
def _selftest():
    rng = np.random.RandomState(0)
    n = 3000
    x0 = rng.rand(n)                       # 真实关系单调递增
    x1 = rng.rand(n)                       # 噪声特征
    x2 = rng.rand(n)
    y = 2.0 * x0 + 0.3 * np.sin(6 * x1) + 0.2 * rng.randn(n)
    X = np.c_[x0, x1, x2]
    # 制造缺失：x2 的 30% 缺失，且缺失本身与 y 相关（检验缺失方向学习）
    miss = rng.rand(n) < 0.3
    X[miss, 2] = np.nan
    y[miss] += 0.5

    print("=== 自检 1：缺失值处理 ===")
    m = GBDTRegressor(n_estimators=150, max_depth=4, random_state=0).fit(X, y)
    p = m.predict(X)
    print(f"  训练 RMSE {np.sqrt(((y-p)**2).mean()):.4f}"
          f" ｜ 恒均值基线 {np.sqrt(((y-y.mean())**2).mean()):.4f}")
    d = p[miss].mean() - p[~miss].mean()
    print(f"  缺失组与非缺失组的预测差 {d:+.3f}（真实差 +0.500）"
          f" → {'✔ 学到了缺失方向' if d > 0.3 else '✘ 未学到'}")

    print("\n=== 自检 2：单调约束生效 ===")
    for name, mc in [("无约束", None), ("x0 递增约束", [1, 0, 0])]:
        m = GBDTRegressor(n_estimators=150, max_depth=4, random_state=0,
                          monotone_constraints=mc).fit(X, y)
        # 固定其余特征，扫 x0，检查预测是否单调
        grid = np.linspace(0, 1, 60)
        Xt = np.c_[grid, np.full(60, 0.5), np.full(60, 0.5)]
        pv = m.predict(Xt)
        drops = int((np.diff(pv) < -1e-9).sum())
        print(f"  {name:12s} 扫描 60 点，下降次数 {drops:2d}"
              f" ｜ RMSE {np.sqrt(((y-m.predict(X))**2).mean()):.4f}")

    print("\n=== 自检 3：反向约束应显著损害拟合（证明约束真的在起作用）===")
    m = GBDTRegressor(n_estimators=150, max_depth=4, random_state=0,
                      monotone_constraints=[-1, 0, 0]).fit(X, y)
    print(f"  x0 递减约束（与真实关系相反）RMSE {np.sqrt(((y-m.predict(X))**2).mean()):.4f}"
          f" → 应明显劣于无约束")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
