"""Equivalence + parity tests: morie.fn._ml_core vs sklearn."""

from morie.fn import _array_core as np
import pytest

pytest.importorskip("sklearn",
                    reason="equivalence baseline needs sklearn")
from sklearn import ensemble as skens
from sklearn import linear_model as sklin
from sklearn.cluster import DBSCAN as SKDB
from sklearn.cluster import KMeans as SKKM
from sklearn.decomposition import PCA as SKPCA
from sklearn.isotonic import IsotonicRegression as SKIso
from sklearn.metrics import r2_score as sk_r2
from sklearn.metrics import roc_auc_score as sk_auc
from sklearn.preprocessing import StandardScaler as SKSS

from morie.fn import _ml_core as ml


def _data():
    rng = np.random.default_rng(23)
    n = 200
    X = rng.normal(0, 1, (n, 4))
    yl = 1.5 + X @ [2.0, -1.0, 0.5, 0.0] + rng.normal(0, 0.5, n)
    pb = 1 / (1 + np.exp(-(0.5 + 1.5 * X[:, 0] - X[:, 1])))
    yb = (rng.uniform(0, 1, n) < pb).astype(int)
    return rng, X, yl, yb


def test_linear_and_ridge_exact():
    _, X, yl, _ = _data()
    g = ml.LinearRegression().fit(X.tolist(), yl.tolist())
    w = sklin.LinearRegression().fit(X, yl)
    assert g.intercept_ == pytest.approx(w.intercept_, rel=1e-9)
    assert list(g.coef_._flat()) == pytest.approx(list(w.coef_),
                                                  rel=1e-9)
    g = ml.Ridge(alpha=2.0).fit(X.tolist(), yl.tolist())
    w = sklin.Ridge(alpha=2.0).fit(X, yl)
    assert list(g.coef_._flat()) == pytest.approx(list(w.coef_),
                                                  rel=1e-8)
    gcv = ml.RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X.tolist(),
                                                  yl.tolist())
    wcv = sklin.RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, yl)
    assert gcv.alpha_ == wcv.alpha_


def test_logistic_same_l2_objective():
    _, X, _, yb = _data()
    g = ml.LogisticRegression(C=1.0).fit(X.tolist(), yb.tolist())
    w = sklin.LogisticRegression(C=1.0).fit(X, yb)
    assert g.coef_.tolist()[0] == pytest.approx(list(w.coef_[0]),
                                                rel=5e-3, abs=5e-3)
    assert g.score(X.tolist(), yb.tolist()) == pytest.approx(
        w.score(X, yb), abs=0.03)


def test_scaler_pca_isotonic_exact():
    rng, X, _, _ = _data()
    g = ml.StandardScaler().fit(X.tolist())
    w = SKSS().fit(X)
    assert g.mean_ == pytest.approx(list(w.mean_), rel=1e-12)
    assert g.scale_ == pytest.approx(list(w.scale_), rel=1e-12)
    gp = ml.PCA(n_components=2).fit(X.tolist())
    wp = SKPCA(n_components=2).fit(X)
    assert gp.explained_variance_ratio_.tolist() == pytest.approx(
        list(wp.explained_variance_ratio_), rel=1e-9)
    xs = rng.uniform(0, 10, 60)
    ys = xs * 0.5 + rng.normal(0, 1, 60)
    gi = ml.IsotonicRegression().fit(xs.tolist(), ys.tolist())
    wi = SKIso().fit(xs, ys)
    q = np.sort(xs)
    assert gi.predict(q.tolist()).tolist() == pytest.approx(
        wi.predict(q).tolist(), rel=1e-9)


def test_forests_and_boosting_parity():
    _, X, yl, yb = _data()
    Xtr, Xte = X[:150], X[150:]
    ytr, yte = yl[:150], yl[150:]
    btr, bte = yb[:150], yb[150:]
    g = ml.RandomForestRegressor(n_estimators=50, random_state=1
                                 ).fit(Xtr.tolist(), ytr.tolist())
    w = skens.RandomForestRegressor(n_estimators=50, random_state=1
                                    ).fit(Xtr, ytr)
    r2g = sk_r2(yte, g.predict(Xte.tolist()).tolist())
    r2w = sk_r2(yte, w.predict(Xte))
    assert r2g > 0.7 and abs(r2g - r2w) < 0.12
    g = ml.RandomForestClassifier(n_estimators=50, random_state=1
                                  ).fit(Xtr.tolist(), btr.tolist())
    w = skens.RandomForestClassifier(n_estimators=50, random_state=1
                                     ).fit(Xtr, btr)
    accg = np.mean(np.array(g.predict(Xte.tolist())) == bte)
    assert accg > 0.65 and abs(accg - w.score(Xte, bte)) < 0.12
    g = ml.GradientBoostingRegressor(n_estimators=60
                                     ).fit(Xtr.tolist(), ytr.tolist())
    w = skens.GradientBoostingRegressor(n_estimators=60).fit(Xtr, ytr)
    r2g = sk_r2(yte, g.predict(Xte.tolist()).tolist())
    assert r2g > 0.8 and abs(r2g - sk_r2(yte, w.predict(Xte))) < 0.1
    g = ml.GradientBoostingClassifier(n_estimators=60
                                      ).fit(Xtr.tolist(), btr.tolist())
    assert np.mean(np.array(g.predict(Xte.tolist())) == bte) > 0.65


def test_clustering_identical_on_blobs():
    from itertools import permutations
    rng = np.random.default_rng(23)
    B = np.vstack([rng.normal(0, .5, (40, 2)),
                   rng.normal(6, .5, (40, 2)),
                   rng.normal([0, 6], .5, (40, 2))])
    g = ml.KMeans(n_clusters=3, n_init=5, random_state=2).fit(B.tolist())
    w = SKKM(n_clusters=3, n_init=5, random_state=2).fit(B)
    lg = [int(v) for v in g.labels_.tolist()]
    lw = w.labels_.tolist()
    match = max(sum(1 for a, b in zip(lg, lw) if p[a] == b)
                for p in permutations(range(3)))
    assert match == len(lg)
    gd = ml.DBSCAN(eps=1.2, min_samples=4).fit(B.tolist())
    wd = SKDB(eps=1.2, min_samples=4).fit(B)
    ng = len(set(int(v) for v in gd.labels_.tolist() if v >= 0))
    nw = len(set(v for v in wd.labels_.tolist() if v >= 0))
    assert ng == nw == 3


def test_svm_parity_and_metrics():
    rng, X, _, yb = _data()
    Xtr, Xte = X[:150], X[150:]
    btr, bte = yb[:150], yb[150:]
    g = ml.SVC(C=1.0).fit(Xtr.tolist(), btr.tolist())
    accg = np.mean(np.array(g.predict(Xte.tolist())) == bte)
    assert accg > 0.6
    scores = rng.uniform(0, 1, len(yb))
    assert ml.roc_auc_score(yb.tolist(), scores.tolist()) == \
        pytest.approx(sk_auc(yb, scores), rel=1e-10)
    rep = ml.classification_report(bte.tolist(),
                                   g.predict(Xte.tolist()),
                                   output_dict=True)
    assert rep["accuracy"] == pytest.approx(accg)
