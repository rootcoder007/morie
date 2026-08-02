"""morie ml core: sklearn subset.

Native replacements for the sklearn surface morie uses: linear_model
(LinearRegression, LogisticRegression with L2, Ridge, RidgeCV),
ensemble (RandomForest/GradientBoosting classifier+regressor), tree
(DecisionTree), preprocessing (StandardScaler, LabelEncoder,
PolynomialFeatures), cluster (KMeans, DBSCAN), decomposition (PCA),
isotonic (IsotonicRegression via PAVA), svm (LinearSVC / SVC / SVR),
metrics (accuracy, roc_auc, classification_report, get_scorer).

Deterministic components are equivalence-tested against sklearn
numerically; randomized ensembles are verified by predictive parity
(accuracy / R^2 within a small delta of sklearn on held-out data) plus
exact-recovery sanity checks — see tests/fn/test_ml_core.py.
"""

from __future__ import annotations

import builtins as _bi
import math as _math

from . import _array_core as _ac


def _X2d(X):
    if hasattr(X, "_cols"):            # native DataFrame
        cols = list(X._cols.values())
        return [[float(c[i]) for c in cols]
                for i in range(X.shape[0])]
    if hasattr(X, "columns") and hasattr(X, "values"):  # real pandas
        return [[float(v) for v in row] for row in X.values.tolist()]
    if hasattr(X, "_data") and hasattr(X, "index"):  # native Series
        return [[float(v)] for v in X._data]
    a = _ac.atleast_2d(X)
    return [list(map(float, r)) for r in a.data]


def _y1d(y):
    if hasattr(y, "_data") and hasattr(y, "index"):  # native Series
        return [float(v) for v in y._data]
    if hasattr(y, "values") and hasattr(y, "index") \
            and not hasattr(y, "_flat"):             # real pandas
        return [float(v) for v in y.values.tolist()]
    return [float(v) for v in _ac.asarray(y)._flat()]


# ===================================================== preprocessing

class StandardScaler:
    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        del y
        Xd = _X2d(X)
        n, d = len(Xd), len(Xd[0])
        self.mean_ = [_math.fsum(Xd[r][j] for r in range(n)) / n
                      for j in range(d)]
        self.var_ = [_math.fsum((Xd[r][j] - self.mean_[j]) ** 2
                                for r in range(n)) / n
                     for j in range(d)]
        self.scale_ = [_math.sqrt(v) if v > 0 else 1.0
                       for v in self.var_]
        return self

    def transform(self, X):
        Xd = _X2d(X)
        out = []
        for r in Xd:
            row = []
            for j, v in enumerate(r):
                u = v - self.mean_[j] if self.with_mean else v
                if self.with_std:
                    u /= self.scale_[j]
                row.append(u)
            out.append(row)
        return _ac.marr(out)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        Xd = _X2d(X)
        return _ac.marr([[v * self.scale_[j] + self.mean_[j]
                          for j, v in enumerate(r)] for r in Xd])


class LabelEncoder:
    def fit(self, y):
        vals = list(y.tolist() if hasattr(y, "tolist") else y)
        self.classes_ = sorted(set(vals), key=str)
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        vals = list(y.tolist() if hasattr(y, "tolist") else y)
        return _ac.marr([float(self._map[v]) for v in vals])

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, idx):
        return [self.classes_[int(i)] for i in
                (idx._flat() if hasattr(idx, "_flat") else idx)]


class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True,
                 interaction_only=False):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only

    def fit(self, X, y=None):
        del y
        self._d = len(_X2d(X)[0])
        return self

    def _combos(self):
        import itertools
        d = self._d
        out = []
        if self.include_bias:
            out.append(())
        for deg in range(1, self.degree + 1):
            gen = (itertools.combinations(range(d), deg)
                   if self.interaction_only else
                   itertools.combinations_with_replacement(
                       range(d), deg))
            out.extend(gen)
        return out

    def transform(self, X):
        Xd = _X2d(X)
        combos = self._combos()
        out = []
        for r in Xd:
            row = []
            for cmb in combos:
                v = 1.0
                for j in cmb:
                    v *= r[j]
                row.append(v)
            out.append(row)
        return _ac.marr(out)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# ===================================================== linear models

class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        Xd = _X2d(X)
        yv = _y1d(y)
        if self.fit_intercept:
            Xd = [[1.0] + r for r in Xd]
        n, k = len(Xd), len(Xd[0])
        XtX = [[_math.fsum(Xd[r][i] * Xd[r][j] for r in range(n))
                for j in range(k)] for i in range(k)]
        Xty = [_math.fsum(Xd[r][i] * yv[r] for r in range(n))
               for i in range(k)]
        b = list(_ac.linalg.solve(_ac.marr(XtX),
                                  _ac.marr(Xty))._flat())
        if self.fit_intercept:
            self.intercept_ = b[0]
            self.coef_ = _ac.marr(b[1:])
        else:
            self.intercept_ = 0.0
            self.coef_ = _ac.marr(b)
        return self

    def predict(self, X):
        Xd = _X2d(X)
        c = list(self.coef_._flat())
        return _ac.marr([self.intercept_
                         + _math.fsum(r[j] * c[j]
                                      for j in range(len(c)))
                         for r in Xd])

    def score(self, X, y):
        yv = _y1d(y)
        p = list(self.predict(X)._flat())
        ybar = _math.fsum(yv) / len(yv)
        ssr = _math.fsum((a - b) ** 2 for a, b in zip(yv, p))
        tss = _math.fsum((a - ybar) ** 2 for a in yv)
        return 1.0 - ssr / tss if tss > 0 else 0.0


class Ridge(LinearRegression):
    def __init__(self, alpha=1.0, fit_intercept=True):
        super().__init__(fit_intercept)
        self.alpha = alpha

    def fit(self, X, y):
        Xd = _X2d(X)
        yv = _y1d(y)
        n = len(Xd)
        d = len(Xd[0])
        if self.fit_intercept:
            # center (sklearn does not penalize the intercept)
            xm = [_math.fsum(Xd[r][j] for r in range(n)) / n
                  for j in range(d)]
            ym = _math.fsum(yv) / n
            Xc = [[Xd[r][j] - xm[j] for j in range(d)]
                  for r in range(n)]
            yc = [v - ym for v in yv]
        else:
            Xc, yc = Xd, yv
        A = [[_math.fsum(Xc[r][i] * Xc[r][j] for r in range(n))
              + (self.alpha if i == j else 0.0)
              for j in range(d)] for i in range(d)]
        b = [_math.fsum(Xc[r][i] * yc[r] for r in range(n))
             for i in range(d)]
        coef = list(_ac.linalg.solve(_ac.marr(A),
                                     _ac.marr(b))._flat())
        self.coef_ = _ac.marr(coef)
        self.intercept_ = (ym - _math.fsum(coef[j] * xm[j]
                                           for j in range(d))) \
            if self.fit_intercept else 0.0
        return self


class RidgeCV(Ridge):
    def __init__(self, alphas=(0.1, 1.0, 10.0), fit_intercept=True):
        super().__init__(1.0, fit_intercept)
        self.alphas = [float(a) for a in alphas]

    def fit(self, X, y):
        Xd = _X2d(X)
        yv = _y1d(y)
        n = len(Xd)
        best, best_alpha = None, self.alphas[0]
        for alpha in self.alphas:
            # leave-one-out via hat matrix (efficient GCV form)
            r = Ridge(alpha, self.fit_intercept).fit(Xd, yv)
            pred = list(r.predict(Xd)._flat())
            press = 0.0
            # hat diagonal from centered design
            d = len(Xd[0])
            xm = [_math.fsum(Xd[t][j] for t in range(n)) / n
                  for j in range(d)]
            Xc = [[Xd[t][j] - xm[j] for j in range(d)]
                  for t in range(n)]
            A = [[_math.fsum(Xc[t][i] * Xc[t][j] for t in range(n))
                  + (alpha if i == j else 0.0)
                  for j in range(d)] for i in range(d)]
            Ainv = _ac.linalg.inv(_ac.marr(A)).tolist()
            for t in range(n):
                h = 1.0 / n + _math.fsum(
                    Xc[t][i] * Ainv[i][j] * Xc[t][j]
                    for i in range(d) for j in range(d))
                e = (yv[t] - pred[t]) / _bi.max(1.0 - h, 1e-10)
                press += e * e
            if best is None or press < best:
                best, best_alpha = press, alpha
        self.alpha_ = best_alpha
        self.alpha = best_alpha
        return Ridge.fit(self, Xd, yv)


class LogisticRegression:
    """Binary logistic with L2 (matches sklearn C parametrization)."""

    def __init__(self, C=1.0, fit_intercept=True, penalty="l2",
                 max_iter=200, **kw):
        del kw
        self.C = C
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.max_iter = max_iter

    def fit(self, X, y):
        Xd = _X2d(X)
        yraw = list(y.tolist() if hasattr(y, "tolist") else y)
        self.classes_ = sorted(set(yraw), key=str)
        if len(self.classes_) != 2:
            raise ValueError("binary only in native core")
        yv = [1.0 if v == self.classes_[1] else 0.0 for v in yraw]
        if self.fit_intercept:
            Xd = [[1.0] + r for r in Xd]
        n, k = len(Xd), len(Xd[0])
        lam = 0.0 if self.penalty in (None, "none") else 1.0 / self.C
        b = [0.0] * k
        for _ in range(self.max_iter):
            eta = [_math.fsum(Xd[r][j] * b[j] for j in range(k))
                   for r in range(n)]
            p = [1.0 / (1.0 + _math.exp(-e)) if e > -30 else 0.0
                 for e in eta]
            g = [_math.fsum(Xd[r][i] * (p[r] - yv[r])
                            for r in range(n)) for i in range(k)]
            H = [[_math.fsum(Xd[r][i] * p[r] * (1.0 - p[r])
                             * Xd[r][j] for r in range(n))
                  for j in range(k)] for i in range(k)]
            for i in range(k):
                if not (self.fit_intercept and i == 0):
                    g[i] += lam * b[i]
                    H[i][i] += lam
                H[i][i] += 1e-10
            step = list(_ac.linalg.solve(_ac.marr(H),
                                         _ac.marr(g))._flat())
            b = [b[i] - step[i] for i in range(k)]
            if max(abs(s) for s in step) < 1e-10:
                break
        if self.fit_intercept:
            self.intercept_ = _ac.marr([b[0]])
            self.coef_ = _ac.marr([b[1:]])
        else:
            self.intercept_ = _ac.marr([0.0])
            self.coef_ = _ac.marr([b])
        return self

    def decision_function(self, X):
        Xd = _X2d(X)
        c = self.coef_.tolist()[0]
        b0 = self.intercept_.tolist()[0]
        return _ac.marr([b0 + _math.fsum(r[j] * c[j]
                                         for j in range(len(c)))
                         for r in Xd])

    def predict_proba(self, X):
        z = self.decision_function(X)._flat()
        out = []
        for e in z:
            p1 = 1.0 / (1.0 + _math.exp(-e)) if e > -30 else 0.0
            out.append([1.0 - p1, p1])
        return _ac.marr(out)

    def predict(self, X):
        return [self.classes_[1] if p[1] >= 0.5 else self.classes_[0]
                for p in self.predict_proba(X).data]

    def score(self, X, y):
        yv = list(y.tolist() if hasattr(y, "tolist") else y)
        p = self.predict(X)
        return _math.fsum(1.0 for a, b in zip(p, yv) if a == b) \
            / len(yv)


# ===================================================== trees

class _Tree:
    __slots__ = ("feat", "thr", "left", "right", "value")

    def __init__(self, value=None):
        self.feat = -1
        self.thr = 0.0
        self.left = None
        self.right = None
        self.value = value


def _build_tree(Xd, yv, idx, depth, max_depth, min_samples_split,
                max_features, rng, classify, n_classes):
    node = _Tree()
    n = len(idx)
    if classify:
        counts = [0.0] * n_classes
        for i in idx:
            counts[int(yv[i])] += 1.0
        node.value = counts
        impurity = 1.0 - _math.fsum((c / n) ** 2 for c in counts)
        pure = impurity <= 0.0
    else:
        m = _math.fsum(yv[i] for i in idx) / n
        node.value = m
        var = _math.fsum((yv[i] - m) ** 2 for i in idx)
        pure = var <= 1e-12
    if (pure or n < min_samples_split
            or (max_depth is not None and depth >= max_depth)):
        return node
    d = len(Xd[0])
    feats = list(range(d))
    if max_features is not None and max_features < d:
        if rng is not None:
            rng.shuffle(feats)
        feats = feats[:max_features]
    best = None
    for f in feats:
        vals = sorted(set(Xd[i][f] for i in idx))
        if len(vals) < 2:
            continue
        order = sorted(idx, key=lambda i: Xd[i][f])
        if classify:
            lc = [0.0] * n_classes
            rc = list(node.value)
            for pos in range(n - 1):
                c = int(yv[order[pos]])
                lc[c] += 1.0
                rc[c] -= 1.0
                if Xd[order[pos]][f] == Xd[order[pos + 1]][f]:
                    continue
                nl, nr = pos + 1.0, n - pos - 1.0
                gl = 1.0 - _math.fsum((v / nl) ** 2 for v in lc)
                gr = 1.0 - _math.fsum((v / nr) ** 2 for v in rc)
                score = (nl * gl + nr * gr) / n
                if best is None or score < best[0]:
                    thr = 0.5 * (Xd[order[pos]][f]
                                 + Xd[order[pos + 1]][f])
                    best = (score, f, thr)
        else:
            sl = 0.0
            sl2 = 0.0
            sr = _math.fsum(yv[i] for i in idx)
            sr2 = _math.fsum(yv[i] ** 2 for i in idx)
            for pos in range(n - 1):
                v = yv[order[pos]]
                sl += v
                sl2 += v * v
                sr -= v
                sr2 -= v * v
                if Xd[order[pos]][f] == Xd[order[pos + 1]][f]:
                    continue
                nl, nr = pos + 1.0, n - pos - 1.0
                score = (sl2 - sl * sl / nl) + (sr2 - sr * sr / nr)
                if best is None or score < best[0]:
                    thr = 0.5 * (Xd[order[pos]][f]
                                 + Xd[order[pos + 1]][f])
                    best = (score, f, thr)
    if best is None:
        return node
    _, f, thr = best
    li = [i for i in idx if Xd[i][f] <= thr]
    ri = [i for i in idx if Xd[i][f] > thr]
    if not li or not ri:
        return node
    node.feat = f
    node.thr = thr
    node.left = _build_tree(Xd, yv, li, depth + 1, max_depth,
                            min_samples_split, max_features, rng,
                            classify, n_classes)
    node.right = _build_tree(Xd, yv, ri, depth + 1, max_depth,
                             min_samples_split, max_features, rng,
                             classify, n_classes)
    return node


def _tree_predict(node, row):
    while node.feat >= 0:
        node = node.left if row[node.feat] <= node.thr \
            else node.right
    return node.value


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2,
                 random_state=None, **kw):
        del kw
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(self, X, y):
        Xd = _X2d(X)
        yv = _y1d(y)
        self._root = _build_tree(Xd, yv, list(range(len(yv))), 0,
                                 self.max_depth,
                                 self.min_samples_split, None, None,
                                 False, 0)
        return self

    def predict(self, X):
        return _ac.marr([_tree_predict(self._root, r)
                         for r in _X2d(X)])


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2,
                 random_state=None, **kw):
        del kw
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(self, X, y):
        Xd = _X2d(X)
        yraw = list(y.tolist() if hasattr(y, "tolist") else y)
        self.classes_ = sorted(set(yraw), key=str)
        cmap = {c: i for i, c in enumerate(self.classes_)}
        yv = [float(cmap[v]) for v in yraw]
        self._root = _build_tree(Xd, yv, list(range(len(yv))), 0,
                                 self.max_depth,
                                 self.min_samples_split, None, None,
                                 True, len(self.classes_))
        return self

    def predict_proba(self, X):
        out = []
        for r in _X2d(X):
            counts = _tree_predict(self._root, r)
            tot = _math.fsum(counts)
            out.append([c / tot for c in counts])
        return _ac.marr(out)

    def predict(self, X):
        out = []
        for p in self.predict_proba(X).data:
            out.append(self.classes_[max(range(len(p)),
                                         key=lambda i: p[i])])
        return out

    def score(self, X, y):
        yv = list(y.tolist() if hasattr(y, "tolist") else y)
        p = self.predict(X)
        return _math.fsum(1.0 for a, b in zip(p, yv) if a == b) \
            / len(yv)


# ===================================================== ensembles

class _ForestBase:
    def __init__(self, n_estimators=100, max_depth=None,
                 min_samples_split=2, max_features=None,
                 random_state=0, **kw):
        del kw
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state or 0

    def _fit_forest(self, Xd, yv, classify, n_classes):
        n = len(yv)
        d = len(Xd[0])
        if self.max_features in (None, "auto"):
            mf = _bi.max(1, int(_math.sqrt(d))) if classify else d
        elif self.max_features == "sqrt":
            mf = _bi.max(1, int(_math.sqrt(d)))
        elif self.max_features == "log2":
            mf = _bi.max(1, int(_math.log2(d)))
        elif isinstance(self.max_features, float):
            mf = _bi.max(1, int(self.max_features * d))
        else:
            mf = int(self.max_features)
        rng = _ac.random.default_rng(self.random_state)
        self._trees = []
        for _t in range(self.n_estimators):
            boot = [int(rng.integers(0, n)) for _ in range(n)]
            self._trees.append(_build_tree(
                Xd, yv, boot, 0, self.max_depth,
                self.min_samples_split, mf, rng, classify,
                n_classes))


class RandomForestRegressor(_ForestBase):
    def fit(self, X, y):
        Xd = _X2d(X)
        yv = _y1d(y)
        self._fit_forest(Xd, yv, False, 0)
        return self

    def predict(self, X):
        Xd = _X2d(X)
        out = []
        for r in Xd:
            preds = [_tree_predict(t, r) for t in self._trees]
            out.append(_math.fsum(preds) / len(preds))
        return _ac.marr(out)

    def score(self, X, y):
        yv = _y1d(y)
        p = list(self.predict(X)._flat())
        ybar = _math.fsum(yv) / len(yv)
        ssr = _math.fsum((a - b) ** 2 for a, b in zip(yv, p))
        tss = _math.fsum((a - ybar) ** 2 for a in yv)
        return 1.0 - ssr / tss if tss > 0 else 0.0


class RandomForestClassifier(_ForestBase):
    def fit(self, X, y):
        Xd = _X2d(X)
        yraw = list(y.tolist() if hasattr(y, "tolist") else y)
        self.classes_ = sorted(set(yraw), key=str)
        cmap = {c: i for i, c in enumerate(self.classes_)}
        yv = [float(cmap[v]) for v in yraw]
        self._fit_forest(Xd, yv, True, len(self.classes_))
        return self

    def predict_proba(self, X):
        Xd = _X2d(X)
        k = len(self.classes_)
        out = []
        for r in Xd:
            acc = [0.0] * k
            for t in self._trees:
                counts = _tree_predict(t, r)
                tot = _math.fsum(counts)
                for i in range(k):
                    acc[i] += counts[i] / tot
            out.append([v / len(self._trees) for v in acc])
        return _ac.marr(out)

    def predict(self, X):
        return [self.classes_[max(range(len(p)),
                                  key=lambda i: p[i])]
                for p in self.predict_proba(X).data]

    def score(self, X, y):
        yv = list(y.tolist() if hasattr(y, "tolist") else y)
        p = self.predict(X)
        return _math.fsum(1.0 for a, b in zip(p, yv) if a == b) \
            / len(yv)

    @property
    def feature_importances_(self):
        # frequency-weighted split counts (proxy importance)
        d_counts = {}

        def walk(node, w):
            if node.feat < 0:
                return
            d_counts[node.feat] = d_counts.get(node.feat, 0.0) + w
            walk(node.left, w)
            walk(node.right, w)
        for t in self._trees:
            walk(t, 1.0)
        tot = _math.fsum(d_counts.values()) or 1.0
        d = max(d_counts) + 1 if d_counts else 0
        return _ac.marr([d_counts.get(j, 0.0) / tot
                         for j in range(d)])


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=0, **kw):
        del kw
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        Xd = _X2d(X)
        yv = _y1d(y)
        n = len(yv)
        self._f0 = _math.fsum(yv) / n
        pred = [self._f0] * n
        self._trees = []
        for _ in range(self.n_estimators):
            resid = [yv[i] - pred[i] for i in range(n)]
            t = _build_tree(Xd, resid, list(range(n)), 0,
                            self.max_depth, 2, None, None, False, 0)
            self._trees.append(t)
            for i in range(n):
                pred[i] += self.learning_rate * _tree_predict(
                    t, Xd[i])
        return self

    def predict(self, X):
        Xd = _X2d(X)
        out = []
        for r in Xd:
            v = self._f0
            for t in self._trees:
                v += self.learning_rate * _tree_predict(t, r)
            out.append(v)
        return _ac.marr(out)

    def score(self, X, y):
        yv = _y1d(y)
        p = list(self.predict(X)._flat())
        ybar = _math.fsum(yv) / len(yv)
        ssr = _math.fsum((a - b) ** 2 for a, b in zip(yv, p))
        tss = _math.fsum((a - ybar) ** 2 for a in yv)
        return 1.0 - ssr / tss if tss > 0 else 0.0


class GradientBoostingClassifier:
    """Binary log-loss boosting on the logit scale."""

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=0, **kw):
        del kw
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y):
        Xd = _X2d(X)
        yraw = list(y.tolist() if hasattr(y, "tolist") else y)
        self.classes_ = sorted(set(yraw), key=str)
        yv = [1.0 if v == self.classes_[1] else 0.0 for v in yraw]
        n = len(yv)
        pbar = _bi.min(_bi.max(_math.fsum(yv) / n, 1e-10),
                       1.0 - 1e-10)
        self._f0 = _math.log(pbar / (1.0 - pbar))
        f = [self._f0] * n
        self._trees = []
        for _ in range(self.n_estimators):
            p = [1.0 / (1.0 + _math.exp(-v)) for v in f]
            resid = [yv[i] - p[i] for i in range(n)]
            t = _build_tree(Xd, resid, list(range(n)), 0,
                            self.max_depth, 2, None, None, False, 0)
            self._trees.append(t)
            for i in range(n):
                f[i] += self.learning_rate * _tree_predict(t, Xd[i])
        return self

    def decision_function(self, X):
        Xd = _X2d(X)
        out = []
        for r in Xd:
            v = self._f0
            for t in self._trees:
                v += self.learning_rate * _tree_predict(t, r)
            out.append(v)
        return _ac.marr(out)

    def predict_proba(self, X):
        out = []
        for v in self.decision_function(X)._flat():
            p1 = 1.0 / (1.0 + _math.exp(-v))
            out.append([1.0 - p1, p1])
        return _ac.marr(out)

    def predict(self, X):
        return [self.classes_[1] if p[1] >= 0.5 else self.classes_[0]
                for p in self.predict_proba(X).data]

    def score(self, X, y):
        yv = list(y.tolist() if hasattr(y, "tolist") else y)
        p = self.predict(X)
        return _math.fsum(1.0 for a, b in zip(p, yv) if a == b) \
            / len(yv)


# ===================================================== clustering / PCA

class KMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300,
                 random_state=0, **kw):
        del kw
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state or 0

    def fit(self, X, y=None):
        del y
        Xd = _X2d(X)
        n, d = len(Xd), len(Xd[0])
        k = self.n_clusters
        best = None
        for init in range(self.n_init):
            rng = _ac.random.default_rng(self.random_state + init)
            # k-means++ seeding
            cents = [list(Xd[int(rng.integers(0, n))])]
            while len(cents) < k:
                dists = []
                for r in Xd:
                    dmin = min(_math.fsum((r[j] - c[j]) ** 2
                                          for j in range(d))
                               for c in cents)
                    dists.append(dmin)
                tot = _math.fsum(dists)
                u = rng.uniform() * tot
                acc = 0.0
                for i, dv in enumerate(dists):
                    acc += dv
                    if acc >= u:
                        cents.append(list(Xd[i]))
                        break
                else:
                    cents.append(list(Xd[-1]))
            labels = [0] * n
            for _it in range(self.max_iter):
                moved = False
                for i, r in enumerate(Xd):
                    bj = min(range(k), key=lambda j: _math.fsum(
                        (r[t] - cents[j][t]) ** 2 for t in range(d)))
                    if bj != labels[i]:
                        labels[i] = bj
                        moved = True
                for j in range(k):
                    mem = [i for i in range(n) if labels[i] == j]
                    if mem:
                        cents[j] = [
                            _math.fsum(Xd[i][t] for i in mem)
                            / len(mem) for t in range(d)]
                if not moved:
                    break
            inertia = _math.fsum(
                _math.fsum((Xd[i][t] - cents[labels[i]][t]) ** 2
                           for t in range(d)) for i in range(n))
            if best is None or inertia < best[0]:
                best = (inertia, cents, labels)
        self.inertia_, cents, labels = best
        self.cluster_centers_ = _ac.marr(cents)
        self.labels_ = _ac.marr([float(v) for v in labels])
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        Xd = _X2d(X)
        cents = self.cluster_centers_.data
        d = len(Xd[0])
        return _ac.marr([float(min(
            range(len(cents)), key=lambda j: _math.fsum(
                (r[t] - cents[j][t]) ** 2 for t in range(d))))
            for r in Xd])


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, **kw):
        del kw
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X, y=None):
        del y
        Xd = _X2d(X)
        n = len(Xd)
        d = len(Xd[0])
        eps2 = self.eps * self.eps

        def neighbors(i):
            out = []
            for j in range(n):
                if _math.fsum((Xd[i][t] - Xd[j][t]) ** 2
                              for t in range(d)) <= eps2:
                    out.append(j)
            return out
        labels = [None] * n
        cid = 0
        for i in range(n):
            if labels[i] is not None:
                continue
            nb = neighbors(i)
            if len(nb) < self.min_samples:
                labels[i] = -1
                continue
            labels[i] = cid
            seeds = [j for j in nb if j != i]
            while seeds:
                j = seeds.pop()
                if labels[j] == -1:
                    labels[j] = cid
                if labels[j] is not None:
                    continue
                labels[j] = cid
                nb2 = neighbors(j)
                if len(nb2) >= self.min_samples:
                    seeds.extend(t for t in nb2
                                 if labels[t] is None
                                 or labels[t] == -1)
            cid += 1
        self.labels_ = _ac.marr([float(v) for v in labels])
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class PCA:
    def __init__(self, n_components=None, **kw):
        del kw
        self.n_components = n_components

    def fit(self, X, y=None):
        del y
        Xd = _X2d(X)
        n, d = len(Xd), len(Xd[0])
        self.mean_ = [_math.fsum(Xd[r][j] for r in range(n)) / n
                      for j in range(d)]
        Xc = [[Xd[r][j] - self.mean_[j] for j in range(d)]
              for r in range(n)]
        cov = [[_math.fsum(Xc[r][i] * Xc[r][j] for r in range(n))
                / (n - 1) for j in range(d)] for i in range(d)]
        w, V = _ac.linalg.eigh(_ac.marr(cov))
        wl = list(w._flat())
        Vd = V.tolist()
        order = sorted(range(d), key=lambda i: -wl[i])
        k = self.n_components or d
        comps = []
        for c in order[:k]:
            vec = [Vd[i][c] for i in range(d)]
            # sign convention: largest-magnitude element positive
            mi = max(range(d), key=lambda i: abs(vec[i]))
            if vec[mi] < 0:
                vec = [-v for v in vec]
            comps.append(vec)
        self.components_ = _ac.marr(comps)
        self.explained_variance_ = _ac.marr(
            [wl[c] for c in order[:k]])
        tot = _math.fsum(wl)
        self.explained_variance_ratio_ = _ac.marr(
            [wl[c] / tot for c in order[:k]])
        return self

    def transform(self, X):
        Xd = _X2d(X)
        comps = self.components_.data
        return _ac.marr([[_math.fsum(
            (r[j] - self.mean_[j]) * comps[c][j]
            for j in range(len(r))) for c in range(len(comps))]
            for r in Xd])

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# ===================================================== isotonic / svm

class IsotonicRegression:
    def __init__(self, increasing=True, **kw):
        del kw
        self.increasing = increasing

    def fit(self, X, y):
        xs = _y1d(X)
        ys = _y1d(y)
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xv = [xs[i] for i in order]
        yv = [ys[i] for i in order]
        if not self.increasing:
            yv = [-v for v in yv]
        # PAVA
        vals = list(yv)
        wts = [1.0] * len(vals)
        blocks = [[i] for i in range(len(vals))]
        i = 0
        while i < len(vals) - 1:
            if vals[i] > vals[i + 1] + 1e-15:
                tot = wts[i] + wts[i + 1]
                merged = (vals[i] * wts[i]
                          + vals[i + 1] * wts[i + 1]) / tot
                vals[i:i + 2] = [merged]
                wts[i:i + 2] = [tot]
                blocks[i:i + 2] = [blocks[i] + blocks[i + 1]]
                if i > 0:
                    i -= 1
            else:
                i += 1
        fitted = [0.0] * len(yv)
        for bval, blk in zip(vals, blocks):
            for j in blk:
                fitted[j] = bval
        if not self.increasing:
            fitted = [-v for v in fitted]
        self._x = xv
        self._y = fitted
        return self

    def predict(self, X):
        import bisect
        xs = _y1d(X)
        out = []
        for v in xs:
            i = bisect.bisect_right(self._x, v) - 1
            i = _bi.max(0, _bi.min(i, len(self._x) - 2))
            x0, x1 = self._x[i], self._x[i + 1]
            y0, y1 = self._y[i], self._y[i + 1]
            if v <= self._x[0]:
                out.append(self._y[0])
            elif v >= self._x[-1]:
                out.append(self._y[-1])
            elif x1 == x0:
                out.append(y0)
            else:
                out.append(y0 + (v - x0) / (x1 - x0) * (y1 - y0))
        return _ac.marr(out)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return _ac.marr(list(self._y))


class LinearSVC:
    """Linear SVM via Pegasos-style subgradient descent."""

    def __init__(self, C=1.0, max_iter=2000, random_state=0, **kw):
        del kw
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state or 0

    def fit(self, X, y):
        Xd = _X2d(X)
        yraw = list(y.tolist() if hasattr(y, "tolist") else y)
        self.classes_ = sorted(set(yraw), key=str)
        ys = [1.0 if v == self.classes_[1] else -1.0 for v in yraw]
        n, d = len(Xd), len(Xd[0])
        lam = 1.0 / (self.C * n)
        w = [0.0] * d
        b = 0.0
        rng = _ac.random.default_rng(self.random_state)
        for t in range(1, self.max_iter + 1):
            i = int(rng.integers(0, n))
            eta = 1.0 / (lam * t)
            margin = ys[i] * (_math.fsum(
                w[j] * Xd[i][j] for j in range(d)) + b)
            for j in range(d):
                w[j] *= (1.0 - eta * lam)
            if margin < 1.0:
                for j in range(d):
                    w[j] += eta * ys[i] * Xd[i][j]
                b += eta * ys[i]
        self.coef_ = _ac.marr([w])
        self.intercept_ = _ac.marr([b])
        return self

    def decision_function(self, X):
        Xd = _X2d(X)
        w = self.coef_.tolist()[0]
        b = self.intercept_.tolist()[0]
        return _ac.marr([b + _math.fsum(r[j] * w[j]
                                        for j in range(len(w)))
                         for r in Xd])

    def predict(self, X):
        return [self.classes_[1] if v >= 0 else self.classes_[0]
                for v in self.decision_function(X)._flat()]

    def score(self, X, y):
        yv = list(y.tolist() if hasattr(y, "tolist") else y)
        p = self.predict(X)
        return _math.fsum(1.0 for a, b in zip(p, yv) if a == b) \
            / len(yv)


class SVC:
    """Kernel SVM via simplified SMO (rbf / linear)."""

    def __init__(self, C=1.0, kernel="rbf", gamma="scale",
                 max_iter=200, probability=False, random_state=0,
                 **kw):
        del kw
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.probability = probability

    def _k(self, a, b):
        if self.kernel == "linear":
            return _math.fsum(x * y for x, y in zip(a, b))
        g = self._gamma
        return _math.exp(-g * _math.fsum((x - y) ** 2
                                         for x, y in zip(a, b)))

    def fit(self, X, y):
        Xd = _X2d(X)
        yraw = list(y.tolist() if hasattr(y, "tolist") else y)
        self.classes_ = sorted(set(yraw), key=str)
        ys = [1.0 if v == self.classes_[1] else -1.0 for v in yraw]
        n, d = len(Xd), len(Xd[0])
        if self.gamma == "scale":
            flat = [v for r in Xd for v in r]
            m = _math.fsum(flat) / len(flat)
            var = _math.fsum((v - m) ** 2 for v in flat) / len(flat)
            self._gamma = 1.0 / (d * var) if var > 0 else 1.0
        elif self.gamma == "auto":
            self._gamma = 1.0 / d
        else:
            self._gamma = float(self.gamma)
        K = [[self._k(Xd[i], Xd[j]) for j in range(n)]
             for i in range(n)]
        alpha = [0.0] * n
        b = 0.0
        C = self.C
        for _sweep in range(self.max_iter):
            changed = 0
            for i in range(n):
                Ei = _math.fsum(alpha[t] * ys[t] * K[t][i]
                                for t in range(n)) + b - ys[i]
                if (ys[i] * Ei < -1e-3 and alpha[i] < C) or \
                        (ys[i] * Ei > 1e-3 and alpha[i] > 0):
                    j = (i + 1 + _sweep) % n
                    if j == i:
                        continue
                    Ej = _math.fsum(alpha[t] * ys[t] * K[t][j]
                                    for t in range(n)) + b - ys[j]
                    ai_old, aj_old = alpha[i], alpha[j]
                    if ys[i] != ys[j]:
                        L = _bi.max(0.0, aj_old - ai_old)
                        H = _bi.min(C, C + aj_old - ai_old)
                    else:
                        L = _bi.max(0.0, ai_old + aj_old - C)
                        H = _bi.min(C, ai_old + aj_old)
                    if L >= H:
                        continue
                    eta = 2.0 * K[i][j] - K[i][i] - K[j][j]
                    if eta >= 0:
                        continue
                    aj = aj_old - ys[j] * (Ei - Ej) / eta
                    aj = _bi.max(L, _bi.min(H, aj))
                    if abs(aj - aj_old) < 1e-6:
                        continue
                    ai = ai_old + ys[i] * ys[j] * (aj_old - aj)
                    alpha[i], alpha[j] = ai, aj
                    b1 = b - Ei - ys[i] * (ai - ai_old) * K[i][i] \
                        - ys[j] * (aj - aj_old) * K[i][j]
                    b2 = b - Ej - ys[i] * (ai - ai_old) * K[i][j] \
                        - ys[j] * (aj - aj_old) * K[j][j]
                    if 0 < ai < C:
                        b = b1
                    elif 0 < aj < C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)
                    changed += 1
            if changed == 0:
                break
        self._sv = [(Xd[i], ys[i], alpha[i]) for i in range(n)
                    if alpha[i] > 1e-10]
        self._b = b
        return self

    def decision_function(self, X):
        Xd = _X2d(X)
        return _ac.marr([
            _math.fsum(a * yv * self._k(sv, r)
                       for sv, yv, a in self._sv) + self._b
            for r in Xd])

    def predict(self, X):
        return [self.classes_[1] if v >= 0 else self.classes_[0]
                for v in self.decision_function(X)._flat()]

    def score(self, X, y):
        yv = list(y.tolist() if hasattr(y, "tolist") else y)
        p = self.predict(X)
        return _math.fsum(1.0 for a, b in zip(p, yv) if a == b) \
            / len(yv)


class SVR(SVC):
    """Epsilon-SVR reduced to kernel ridge for the native core.

    ponytail: kernel ridge minimizes the same RKHS norm with squared
    loss; swap in true eps-insensitive SMO if a caller needs exact
    sparse SVR behavior.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma="scale",
                 epsilon=0.1, **kw):
        super().__init__(C=C, kernel=kernel, gamma=gamma)
        self.epsilon = epsilon

    def fit(self, X, y):
        Xd = _X2d(X)
        yv = _y1d(y)
        n, d = len(Xd), len(Xd[0])
        if self.gamma == "scale":
            flat = [v for r in Xd for v in r]
            m = _math.fsum(flat) / len(flat)
            var = _math.fsum((v - m) ** 2 for v in flat) / len(flat)
            self._gamma = 1.0 / (d * var) if var > 0 else 1.0
        else:
            self._gamma = 1.0 / d if self.gamma == "auto" \
                else float(self.gamma)
        K = [[self._k(Xd[i], Xd[j]) + (1.0 / self.C if i == j
                                       else 0.0)
              for j in range(n)] for i in range(n)]
        alpha = _ac.linalg.solve(_ac.marr(K), _ac.marr(yv))
        self._X = Xd
        self._alpha = list(alpha._flat())
        return self

    def predict(self, X):
        Xd = _X2d(X)
        return _ac.marr([
            _math.fsum(self._alpha[i] * self._k(self._X[i], r)
                       for i in range(len(self._X))) for r in Xd])


# ===================================================== metrics

def accuracy_score(y_true, y_pred):
    yt = list(y_true.tolist() if hasattr(y_true, "tolist")
              else y_true)
    yp = list(y_pred.tolist() if hasattr(y_pred, "tolist")
              else y_pred)
    return _math.fsum(1.0 for a, b in zip(yt, yp) if a == b) \
        / len(yt)


def roc_auc_score(y_true, y_score):
    yt = [float(v) for v in (y_true.tolist()
                             if hasattr(y_true, "tolist")
                             else y_true)]
    ys = [float(v) for v in (y_score.tolist()
                             if hasattr(y_score, "tolist")
                             else y_score)]
    # Mann-Whitney with midranks (exact AUC incl ties)
    from . import _stats_core as _stats
    ranks = _stats.rankdata(ys)
    pos = [i for i, v in enumerate(yt) if v == 1.0]
    n1 = len(pos)
    n0 = len(yt) - n1
    r1 = _math.fsum(ranks[i] for i in pos)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def mean_squared_error(y_true, y_pred):
    yt = _y1d(y_true)
    yp = _y1d(y_pred)
    return _math.fsum((a - b) ** 2 for a, b in zip(yt, yp)) / len(yt)


def mean_absolute_error(y_true, y_pred):
    yt = _y1d(y_true)
    yp = _y1d(y_pred)
    return _math.fsum(abs(a - b) for a, b in zip(yt, yp)) / len(yt)


def r2_score(y_true, y_pred):
    yt = _y1d(y_true)
    yp = _y1d(y_pred)
    ybar = _math.fsum(yt) / len(yt)
    ssr = _math.fsum((a - b) ** 2 for a, b in zip(yt, yp))
    tss = _math.fsum((a - ybar) ** 2 for a in yt)
    return 1.0 - ssr / tss if tss > 0 else 0.0


def precision_recall_f1(y_true, y_pred, positive):
    yt = list(y_true.tolist() if hasattr(y_true, "tolist")
              else y_true)
    yp = list(y_pred.tolist() if hasattr(y_pred, "tolist")
              else y_pred)
    tp = sum(1 for a, b in zip(yt, yp)
             if a == positive and b == positive)
    fp = sum(1 for a, b in zip(yt, yp)
             if a != positive and b == positive)
    fn = sum(1 for a, b in zip(yt, yp)
             if a == positive and b != positive)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def f1_score(y_true, y_pred, pos_label=1):
    return precision_recall_f1(y_true, y_pred, pos_label)[2]


def classification_report(y_true, y_pred, output_dict=False):
    yt = list(y_true.tolist() if hasattr(y_true, "tolist")
              else y_true)
    classes = sorted(set(yt), key=str)
    rep = {}
    for c in classes:
        p, r, f1 = precision_recall_f1(y_true, y_pred, c)
        rep[str(c)] = {"precision": p, "recall": r, "f1-score": f1,
                       "support": sum(1 for v in yt if v == c)}
    rep["accuracy"] = accuracy_score(y_true, y_pred)
    if output_dict:
        return rep
    lines = ["%-10s %9s %9s %9s %9s" % ("", "precision", "recall",
                                        "f1-score", "support")]
    for c in classes:
        d = rep[str(c)]
        lines.append("%-10s %9.3f %9.3f %9.3f %9d" % (
            str(c)[:10], d["precision"], d["recall"], d["f1-score"],
            d["support"]))
    lines.append("accuracy %.3f" % rep["accuracy"])
    return "\n".join(lines)


def get_scorer(name):
    table = {
        "accuracy": lambda est, X, y: est.score(X, y),
        "r2": lambda est, X, y: r2_score(y, est.predict(X)),
        "neg_mean_squared_error": lambda est, X, y:
            -mean_squared_error(y, est.predict(X)),
        "roc_auc": lambda est, X, y: roc_auc_score(
            y, [p[1] for p in est.predict_proba(X).data]),
        "f1": lambda est, X, y: f1_score(y, est.predict(X)),
    }
    if name not in table:
        raise ValueError("unsupported scorer %r" % name)
    return table[name]


# namespace mirrors
class linear_model:
    LinearRegression = LinearRegression
    LogisticRegression = LogisticRegression
    Ridge = Ridge
    RidgeCV = RidgeCV


class ensemble:
    RandomForestClassifier = RandomForestClassifier
    RandomForestRegressor = RandomForestRegressor
    GradientBoostingClassifier = GradientBoostingClassifier
    GradientBoostingRegressor = GradientBoostingRegressor


class tree:
    DecisionTreeClassifier = DecisionTreeClassifier
    DecisionTreeRegressor = DecisionTreeRegressor


class preprocessing:
    StandardScaler = StandardScaler
    LabelEncoder = LabelEncoder
    PolynomialFeatures = PolynomialFeatures


class cluster:
    KMeans = KMeans
    DBSCAN = DBSCAN


class decomposition:
    PCA = PCA


class isotonic:
    IsotonicRegression = IsotonicRegression


class svm:
    SVC = SVC
    SVR = SVR
    LinearSVC = LinearSVC


class metrics:
    accuracy_score = staticmethod(accuracy_score)
    roc_auc_score = staticmethod(roc_auc_score)
    mean_squared_error = staticmethod(mean_squared_error)
    mean_absolute_error = staticmethod(mean_absolute_error)
    r2_score = staticmethod(r2_score)
    f1_score = staticmethod(f1_score)
    classification_report = staticmethod(classification_report)
    get_scorer = staticmethod(get_scorer)


# ===================================================== linear tail

class Lasso(LinearRegression):
    """Coordinate descent on the sklearn objective
    (1/2n)||y-Xb||^2 + alpha*||b||_1."""

    def __init__(self, alpha=1.0, fit_intercept=True,
                 max_iter=2000, tol=1e-8):
        super().__init__(fit_intercept)
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = 1.0

    def fit(self, X, y):
        Xd = _X2d(X)
        yv = _y1d(y)
        n, d = len(Xd), len(Xd[0])
        if self.fit_intercept:
            xm = [_math.fsum(Xd[r][j] for r in range(n)) / n
                  for j in range(d)]
            ym = _math.fsum(yv) / n
            Xc = [[Xd[r][j] - xm[j] for j in range(d)]
                  for r in range(n)]
            yc = [v - ym for v in yv]
        else:
            xm, ym = [0.0] * d, 0.0
            Xc, yc = Xd, yv
        col_ss = [_math.fsum(Xc[r][j] ** 2 for r in range(n))
                  for j in range(d)]
        b = [0.0] * d
        resid = list(yc)
        l1 = self.alpha * self.l1_ratio * n
        l2 = self.alpha * (1.0 - self.l1_ratio) * n
        for _sweep in range(self.max_iter):
            delta = 0.0
            for j in range(d):
                if col_ss[j] == 0.0:
                    continue
                rho = _math.fsum(Xc[r][j] * resid[r]
                                 for r in range(n)) \
                    + b[j] * col_ss[j]
                if rho > l1:
                    new = (rho - l1) / (col_ss[j] + l2)
                elif rho < -l1:
                    new = (rho + l1) / (col_ss[j] + l2)
                else:
                    new = 0.0
                if new != b[j]:
                    diff = new - b[j]
                    for r in range(n):
                        resid[r] -= diff * Xc[r][j]
                    delta = _bi.max(delta, abs(diff))
                    b[j] = new
            if delta < self.tol:
                break
        self.coef_ = _ac.marr(b)
        self.intercept_ = ym - _math.fsum(b[j] * xm[j]
                                          for j in range(d)) \
            if self.fit_intercept else 0.0
        return self


class ElasticNet(Lasso):
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 max_iter=2000, tol=1e-8):
        super().__init__(alpha, fit_intercept, max_iter, tol)
        self.l1_ratio = l1_ratio


class BayesianRidge(LinearRegression):
    """Evidence-maximization (MacKay) alpha/lambda updates."""

    def __init__(self, fit_intercept=True, max_iter=300, tol=1e-6):
        super().__init__(fit_intercept)
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        Xd = _X2d(X)
        yv = _y1d(y)
        n, d = len(Xd), len(Xd[0])
        if self.fit_intercept:
            xm = [_math.fsum(Xd[r][j] for r in range(n)) / n
                  for j in range(d)]
            ym = _math.fsum(yv) / n
            Xc = [[Xd[r][j] - xm[j] for j in range(d)]
                  for r in range(n)]
            yc = [v - ym for v in yv]
        else:
            xm, ym = [0.0] * d, 0.0
            Xc, yc = Xd, yv
        alpha_ = 1.0 / (_math.fsum(v * v for v in yc) / n + 1e-10)
        lambda_ = 1.0
        XtX = [[_math.fsum(Xc[r][i] * Xc[r][j] for r in range(n))
                for j in range(d)] for i in range(d)]
        Xty = [_math.fsum(Xc[r][i] * yc[r] for r in range(n))
               for i in range(d)]
        b = [0.0] * d
        for _ in range(self.max_iter):
            A = [[alpha_ * XtX[i][j] + (lambda_ if i == j else 0.0)
                  for j in range(d)] for i in range(d)]
            rhs = [alpha_ * v for v in Xty]
            bn = list(_ac.linalg.solve(_ac.marr(A),
                                       _ac.marr(rhs))._flat())
            Sinv = _ac.linalg.inv(_ac.marr(A)).tolist()
            gamma_ = d - lambda_ * _math.fsum(Sinv[i][i]
                                              for i in range(d))
            ssb = _math.fsum(v * v for v in bn) + 1e-300
            resid = [yc[r] - _math.fsum(Xc[r][j] * bn[j]
                                        for j in range(d))
                     for r in range(n)]
            ssr = _math.fsum(v * v for v in resid) + 1e-300
            lambda_new = gamma_ / ssb
            alpha_new = (n - gamma_) / ssr
            done = max(abs(a - c) for a, c in zip(bn, b)) < self.tol
            b = bn
            alpha_, lambda_ = alpha_new, lambda_new
            if done:
                break
        self.coef_ = _ac.marr(b)
        self.intercept_ = ym - _math.fsum(b[j] * xm[j]
                                          for j in range(d)) \
            if self.fit_intercept else 0.0
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        return self


linear_model.Lasso = Lasso
linear_model.ElasticNet = ElasticNet
linear_model.BayesianRidge = BayesianRidge


# ===================================================== model_selection

def train_test_split(*arrays, test_size=0.25, random_state=0,
                     shuffle=True, stratify=None):
    del stratify
    n = len(arrays[0].tolist() if hasattr(arrays[0], "tolist")
            else arrays[0])
    idx = list(range(n))
    if shuffle:
        rng = _ac.random.default_rng(random_state or 0)
        rng.shuffle(idx)
    ntest = int(round(n * test_size)) if test_size < 1 \
        else int(test_size)
    te, tr = idx[:ntest], idx[ntest:]
    out = []
    for a in arrays:
        av = a.tolist() if hasattr(a, "tolist") else list(a)
        out.append([av[i] for i in tr])
        out.append([av[i] for i in te])
    return out


class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=0):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state or 0

    def split(self, X, y=None):
        del y
        n = len(X.tolist() if hasattr(X, "tolist") else X)
        idx = list(range(n))
        if self.shuffle:
            rng = _ac.random.default_rng(self.random_state)
            rng.shuffle(idx)
        sizes = [n // self.n_splits
                 + (1 if i < n % self.n_splits else 0)
                 for i in range(self.n_splits)]
        pos = 0
        for s in sizes:
            test = idx[pos:pos + s]
            train = idx[:pos] + idx[pos + s:]
            yield train, test
            pos += s


class StratifiedKFold(KFold):
    def split(self, X, y):
        yv = list(y.tolist() if hasattr(y, "tolist") else y)
        n = len(yv)
        byclass = {}
        for i, v in enumerate(yv):
            byclass.setdefault(v, []).append(i)
        if self.shuffle:
            rng = _ac.random.default_rng(self.random_state)
            for v in byclass:
                rng.shuffle(byclass[v])
        folds = [[] for _ in range(self.n_splits)]
        for v, members in byclass.items():
            for k, i in enumerate(members):
                folds[k % self.n_splits].append(i)
        for k in range(self.n_splits):
            test = sorted(folds[k])
            train = sorted(i for i in range(n) if i not in set(test))
            yield train, test


def _index_rows(X, idx):
    Xv = X.tolist() if hasattr(X, "tolist") else list(X)
    return [Xv[i] for i in idx]


def cross_val_score(estimator, X, y, cv=5, scoring=None):
    import copy
    folds = cv if hasattr(cv, "split") else KFold(n_splits=cv)
    scorer = get_scorer(scoring) if isinstance(scoring, str) else None
    scores = []
    for tr, te in folds.split(X, y):
        est = copy.deepcopy(estimator)
        est.fit(_index_rows(X, tr), _index_rows(y, tr))
        if scorer is not None:
            scores.append(scorer(est, _index_rows(X, te),
                                 _index_rows(y, te)))
        else:
            scores.append(est.score(_index_rows(X, te),
                                    _index_rows(y, te)))
    return _ac.marr(scores)


class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, scoring=None,
                 **kw):
        del kw
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def _grid(self):
        import itertools
        keys = list(self.param_grid)
        for combo in itertools.product(*(self.param_grid[k]
                                         for k in keys)):
            yield dict(zip(keys, combo))

    def fit(self, X, y):
        import copy
        best = None
        for params in self._grid():
            est = copy.deepcopy(self.estimator)
            for k, v in params.items():
                setattr(est, k, v)
            sc = cross_val_score(est, X, y, cv=self.cv,
                                 scoring=self.scoring)
            m = _math.fsum(sc._flat()) / len(sc._flat())
            if best is None or m > best[0]:
                best = (m, params)
        self.best_score_, self.best_params_ = best
        self.best_estimator_ = copy.deepcopy(self.estimator)
        for k, v in self.best_params_.items():
            setattr(self.best_estimator_, k, v)
        self.best_estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)


class RandomizedSearchCV(GridSearchCV):
    def __init__(self, estimator, param_distributions, n_iter=10,
                 cv=5, scoring=None, random_state=0, **kw):
        super().__init__(estimator, param_distributions, cv, scoring)
        self.n_iter = n_iter
        self.random_state = random_state or 0

    def _grid(self):
        rng = _ac.random.default_rng(self.random_state)
        keys = list(self.param_grid)
        for _ in range(self.n_iter):
            yield {k: self.param_grid[k][int(rng.integers(
                0, len(self.param_grid[k])))] for k in keys}


def learning_curve(estimator, X, y, train_sizes=(0.1, 0.33, 0.55,
                                                 0.78, 1.0), cv=5,
                   scoring=None):
    import copy
    n = len(X.tolist() if hasattr(X, "tolist") else X)
    folds = list(KFold(n_splits=cv).split(X, y))
    sizes_abs = []
    train_scores = []
    test_scores = []
    scorer = get_scorer(scoring) if isinstance(scoring, str) else None
    for frac in train_sizes:
        tr_scores = []
        te_scores = []
        m = None
        for tr, te in folds:
            m = int(len(tr) * frac) if frac <= 1 else int(frac)
            sub = tr[:m]
            est = copy.deepcopy(estimator)
            est.fit(_index_rows(X, sub), _index_rows(y, sub))
            if scorer is not None:
                tr_scores.append(scorer(est, _index_rows(X, sub),
                                        _index_rows(y, sub)))
                te_scores.append(scorer(est, _index_rows(X, te),
                                        _index_rows(y, te)))
            else:
                tr_scores.append(est.score(_index_rows(X, sub),
                                           _index_rows(y, sub)))
                te_scores.append(est.score(_index_rows(X, te),
                                           _index_rows(y, te)))
        sizes_abs.append(m)
        train_scores.append(tr_scores)
        test_scores.append(te_scores)
    return (_ac.marr([float(s) for s in sizes_abs]),
            _ac.marr(train_scores), _ac.marr(test_scores))


class model_selection:
    train_test_split = staticmethod(train_test_split)
    KFold = KFold
    StratifiedKFold = StratifiedKFold
    cross_val_score = staticmethod(cross_val_score)
    GridSearchCV = GridSearchCV
    RandomizedSearchCV = RandomizedSearchCV
    learning_curve = staticmethod(learning_curve)


# ===================================================== metrics tail

def roc_curve(y_true, y_score):
    yt = [float(v) for v in (y_true.tolist()
                             if hasattr(y_true, "tolist")
                             else y_true)]
    ys = [float(v) for v in (y_score.tolist()
                             if hasattr(y_score, "tolist")
                             else y_score)]
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    P = sum(1 for v in yt if v == 1.0)
    N = len(yt) - P
    fpr = [0.0]
    tpr = [0.0]
    thr = [float("inf")]
    tp = fp = 0
    i = 0
    while i < len(order):
        t = ys[order[i]]
        while i < len(order) and ys[order[i]] == t:
            if yt[order[i]] == 1.0:
                tp += 1
            else:
                fp += 1
            i += 1
        fpr.append(fp / N if N else 0.0)
        tpr.append(tp / P if P else 0.0)
        thr.append(t)
    return _ac.marr(fpr), _ac.marr(tpr), _ac.marr(thr)


def precision_recall_curve(y_true, y_score):
    yt = [float(v) for v in (y_true.tolist()
                             if hasattr(y_true, "tolist")
                             else y_true)]
    ys = [float(v) for v in (y_score.tolist()
                             if hasattr(y_score, "tolist")
                             else y_score)]
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    P = sum(1 for v in yt if v == 1.0)
    prec = []
    rec = []
    thr = []
    tp = fp = 0
    i = 0
    while i < len(order):
        t = ys[order[i]]
        while i < len(order) and ys[order[i]] == t:
            if yt[order[i]] == 1.0:
                tp += 1
            else:
                fp += 1
            i += 1
        prec.append(tp / (tp + fp))
        rec.append(tp / P if P else 0.0)
        thr.append(t)
    prec.append(1.0)
    rec.append(0.0)
    return _ac.marr(prec), _ac.marr(rec), _ac.marr(thr)


def average_precision_score(y_true, y_score):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    pv = list(prec._flat())
    rv = list(rec._flat())
    # curve is built with recall increasing; AP = sum dR * P
    ap = 0.0
    prev_r = 0.0
    for i in range(len(rv) - 1):        # last point is the (1, 0) pad
        ap += (rv[i] - prev_r) * pv[i]
        prev_r = rv[i]
    return ap


def log_loss(y_true, y_prob, eps=1e-15):
    yt = [float(v) for v in (y_true.tolist()
                             if hasattr(y_true, "tolist")
                             else y_true)]
    yp = y_prob.tolist() if hasattr(y_prob, "tolist") else list(y_prob)
    total = 0.0
    for t, p in zip(yt, yp):
        p1 = p[1] if isinstance(p, (list, tuple)) else float(p)
        p1 = _bi.min(_bi.max(p1, eps), 1.0 - eps)
        total += -(t * _math.log(p1) + (1.0 - t) * _math.log(1.0 - p1))
    return total / len(yt)


def confusion_matrix(y_true, y_pred):
    yt = list(y_true.tolist() if hasattr(y_true, "tolist")
              else y_true)
    yp = list(y_pred.tolist() if hasattr(y_pred, "tolist")
              else y_pred)
    classes = sorted(set(yt) | set(yp), key=str)
    cmap = {c: i for i, c in enumerate(classes)}
    m = [[0.0] * len(classes) for _ in classes]
    for a, b in zip(yt, yp):
        m[cmap[a]][cmap[b]] += 1.0
    return _ac.marr(m)


def calibration_curve(y_true, y_prob, n_bins=5):
    yt = [float(v) for v in (y_true.tolist()
                             if hasattr(y_true, "tolist")
                             else y_true)]
    yp = [float(v) for v in (y_prob.tolist()
                             if hasattr(y_prob, "tolist")
                             else y_prob)]
    frac = []
    mean_pred = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        members = [i for i, p in enumerate(yp)
                   if (lo < p <= hi) or (b == 0 and p == 0.0)]
        if not members:
            continue
        frac.append(_math.fsum(yt[i] for i in members)
                    / len(members))
        mean_pred.append(_math.fsum(yp[i] for i in members)
                         / len(members))
    return _ac.marr(frac), _ac.marr(mean_pred)


for _n in ("roc_curve", "precision_recall_curve",
           "average_precision_score", "log_loss",
           "confusion_matrix", "calibration_curve"):
    setattr(metrics, _n, staticmethod(globals()[_n]))


# ===================================================== neighbors / tsne

class NearestNeighbors:
    def __init__(self, n_neighbors=5, **kw):
        del kw
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        del y
        self._X = _X2d(X)
        return self

    def kneighbors(self, X=None, n_neighbors=None):
        q = self._X if X is None else _X2d(X)
        k = n_neighbors or self.n_neighbors
        d = len(self._X[0])
        dists = []
        idxs = []
        for r in q:
            dd = sorted(
                (_math.sqrt(_math.fsum((r[t] - s[t]) ** 2
                                       for t in range(d))), i)
                for i, s in enumerate(self._X))
            if X is None:
                dd = dd[1:]        # exclude self
            dists.append([v for v, _ in dd[:k]])
            idxs.append([float(i) for _, i in dd[:k]])
        return _ac.marr(dists), _ac.marr(idxs)


class TSNE:
    """Exact t-SNE (no Barnes-Hut): fine for the small n morie plots."""

    def __init__(self, n_components=2, perplexity=30.0,
                 learning_rate=200.0, n_iter=500, random_state=0,
                 **kw):
        del kw
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state or 0

    def fit_transform(self, X, y=None):
        del y
        Xd = _X2d(X)
        n = len(Xd)
        d = len(Xd[0])
        # pairwise squared distances
        D = [[_math.fsum((Xd[i][t] - Xd[j][t]) ** 2
                         for t in range(d)) for j in range(n)]
             for i in range(n)]
        # binary-search sigmas for target perplexity
        target = _math.log(_bi.min(self.perplexity, (n - 1) / 3.0))
        P = [[0.0] * n for _ in range(n)]
        for i in range(n):
            lo, hi = 1e-10, 1e10
            beta = 1.0
            for _ in range(60):
                num = [_math.exp(-D[i][j] * beta) if j != i else 0.0
                       for j in range(n)]
                s = _math.fsum(num) + 1e-300
                H = _math.log(s) + beta * _math.fsum(
                    num[j] * D[i][j] for j in range(n)) / s
                if abs(H - target) < 1e-5:
                    break
                if H > target:
                    lo = beta
                    beta = beta * 2 if hi >= 1e10 else 0.5 * (lo + hi)
                else:
                    hi = beta
                    beta = 0.5 * (lo + hi)
            num = [_math.exp(-D[i][j] * beta) if j != i else 0.0
                   for j in range(n)]
            s = _math.fsum(num) + 1e-300
            for j in range(n):
                P[i][j] = num[j] / s
        # symmetrize
        for i in range(n):
            for j in range(i + 1, n):
                v = (P[i][j] + P[j][i]) / (2.0 * n)
                P[i][j] = P[j][i] = _bi.max(v, 1e-12)
        rng = _ac.random.default_rng(self.random_state)
        Y = [[rng.normal(0.0, 1e-4)
              for _ in range(self.n_components)] for _ in range(n)]
        vel = [[0.0] * self.n_components for _ in range(n)]
        for it in range(self.n_iter):
            mom = 0.5 if it < 250 else 0.8
            exag = 12.0 if it < 100 else 1.0
            Q = [[0.0] * n for _ in range(n)]
            qs = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dq = 1.0 / (1.0 + _math.fsum(
                        (Y[i][t] - Y[j][t]) ** 2
                        for t in range(self.n_components)))
                    Q[i][j] = Q[j][i] = dq
                    qs += 2.0 * dq
            for i in range(n):
                grad = [0.0] * self.n_components
                for j in range(n):
                    if j == i:
                        continue
                    coef = 4.0 * (exag * P[i][j] - Q[i][j] / qs) \
                        * Q[i][j]
                    for t in range(self.n_components):
                        grad[t] += coef * (Y[i][t] - Y[j][t])
                for t in range(self.n_components):
                    vel[i][t] = mom * vel[i][t] \
                        - self.learning_rate * grad[t]
                    # clamp step to keep the exact-gradient descent
                    # stable at high learning rates
                    vel[i][t] = _bi.max(-5.0, _bi.min(5.0,
                                                      vel[i][t]))
                    Y[i][t] += vel[i][t]
        return _ac.marr(Y)


class neighbors:
    NearestNeighbors = NearestNeighbors


class manifold:
    TSNE = TSNE


class BaseEstimator:
    def get_params(self, deep=True):
        del deep
        return {k: v for k, v in vars(self).items()
                if not k.endswith("_") and not k.startswith("_")}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def clone(estimator):
    import copy
    new = copy.deepcopy(estimator)
    for k in list(vars(new)):
        if k.endswith("_") and not k.startswith("_"):
            delattr(new, k)
    return new


class base:
    BaseEstimator = BaseEstimator
    clone = staticmethod(clone)


# ===================================================== splitters tail

class GroupKFold:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        del y
        gv = list(groups.tolist() if hasattr(groups, "tolist")
                  else groups)
        uniq = sorted(set(gv), key=str)
        # assign groups to folds by size (largest first, greedy)
        sizes = {g: gv.count(g) for g in uniq}
        folds = [[] for _ in range(self.n_splits)]
        loads = [0] * self.n_splits
        for g in sorted(uniq, key=lambda u: -sizes[u]):
            k = loads.index(min(loads))
            folds[k].append(g)
            loads[k] += sizes[g]
        n = len(gv)
        for k in range(self.n_splits):
            gset = set(folds[k])
            test = [i for i in range(n) if gv[i] in gset]
            train = [i for i in range(n) if gv[i] not in gset]
            yield train, test


class LeaveOneOut:
    def split(self, X, y=None):
        del y
        n = len(X.tolist() if hasattr(X, "tolist") else X)
        for i in range(n):
            yield [j for j in range(n) if j != i], [i]


class ShuffleSplit:
    def __init__(self, n_splits=10, test_size=0.1, random_state=0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state or 0

    def split(self, X, y=None):
        del y
        n = len(X.tolist() if hasattr(X, "tolist") else X)
        ntest = int(round(n * self.test_size)) \
            if self.test_size < 1 else int(self.test_size)
        rng = _ac.random.default_rng(self.random_state)
        for _ in range(self.n_splits):
            idx = list(range(n))
            rng.shuffle(idx)
            yield idx[ntest:], idx[:ntest]


class TimeSeriesSplit:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def split(self, X, y=None):
        del y
        n = len(X.tolist() if hasattr(X, "tolist") else X)
        fold = n // (self.n_splits + 1)
        for k in range(1, self.n_splits + 1):
            train = list(range(0, fold * k))
            test = list(range(fold * k,
                              min(fold * (k + 1), n)))
            yield train, test


for _n in ("GroupKFold", "LeaveOneOut", "ShuffleSplit",
           "TimeSeriesSplit"):
    setattr(model_selection, _n, globals()[_n])


def precision_recall_fscore_support(y_true, y_pred, average=None,
                                    **kw):
    del kw
    yt = list(y_true.tolist() if hasattr(y_true, "tolist")
              else y_true)
    classes = sorted(set(yt), key=str)
    precs, recs, f1s, sups = [], [], [], []
    for c in classes:
        p, r, f1 = precision_recall_f1(y_true, y_pred, c)
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
        sups.append(float(sum(1 for v in yt if v == c)))
    if average == "macro":
        k = len(classes)
        return (_math.fsum(precs) / k, _math.fsum(recs) / k,
                _math.fsum(f1s) / k, None)
    if average == "weighted":
        tot = _math.fsum(sups)
        return (
            _math.fsum(p * s for p, s in zip(precs, sups)) / tot,
            _math.fsum(r * s for r, s in zip(recs, sups)) / tot,
            _math.fsum(f * s for f, s in zip(f1s, sups)) / tot,
            None)
    return (_ac.marr(precs), _ac.marr(recs), _ac.marr(f1s),
            _ac.marr(sups))


metrics.precision_recall_fscore_support = staticmethod(
    precision_recall_fscore_support)


# ===================================================== imputation

def enable_iterative_imputer():
    """No-op: native IterativeImputer is always available."""


class IterativeImputer:
    """Round-robin regression imputation (BayesianRidge per column)."""

    def __init__(self, max_iter=10, random_state=0, tol=1e-3, **kw):
        del kw
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(self, X, y=None):
        del y
        Xd = [[float(v) if v is not None and v == v else None
               for v in row]
              for row in (X.tolist() if hasattr(X, "tolist")
                          else X)]
        n, d = len(Xd), len(Xd[0])
        miss = [(r, c) for r in range(n) for c in range(d)
                if Xd[r][c] is None]
        # initial fill: column means
        means = []
        for c in range(d):
            vals = [Xd[r][c] for r in range(n)
                    if Xd[r][c] is not None]
            means.append(_math.fsum(vals) / len(vals)
                         if vals else 0.0)
        for r, c in miss:
            Xd[r][c] = means[c]
        for _sweep in range(self.max_iter):
            delta = 0.0
            for c in range(d):
                rows_c = [r for r, cc in miss if cc == c]
                if not rows_c:
                    continue
                obs = [r for r in range(n) if r not in set(rows_c)]
                feats = [j for j in range(d) if j != c]
                reg = BayesianRidge().fit(
                    [[Xd[r][j] for j in feats] for r in obs],
                    [Xd[r][c] for r in obs])
                pred = reg.predict(
                    [[Xd[r][j] for j in feats] for r in rows_c])
                for k, r in enumerate(rows_c):
                    new = float(pred[k])
                    delta = _bi.max(delta, abs(new - Xd[r][c]))
                    Xd[r][c] = new
            if delta < self.tol:
                break
        return _ac.marr(Xd)

    def fit(self, X, y=None):
        self._fitted = self.fit_transform(X)
        return self

    def transform(self, X):
        return self.fit_transform(X)


class impute:
    IterativeImputer = IterativeImputer


class experimental:
    enable_iterative_imputer = staticmethod(enable_iterative_imputer)
