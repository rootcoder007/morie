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
    a = _ac.atleast_2d(X)
    return [list(map(float, r)) for r in a.data]


def _y1d(y):
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
