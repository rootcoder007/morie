"""Equivalence + parity tests: morie.fn._ml_core vs frozen sklearn
anchors.

Reference values in oracle_anchors.json were computed ONCE from
sklearn (version recorded in the anchors) on exactly the inputs these
tests regenerate from morie's own RNG (Philox, seed 23). sklearn is
not imported anywhere.
"""
import json
import math
import pathlib

import pytest

from morie.fn import _array_core as np
from morie.fn import _ml_core as ml

A = json.loads(pathlib.Path(__file__).with_name(
    "oracle_anchors.json").read_text())


def _l(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


def _data():
    rng = np.random.default_rng(23)
    n = 200
    X = rng.normal(0, 1, (n, 4))
    Xl = X.tolist()
    eps = [float(v) for v in rng.normal(0, 0.5, n)._flat()]
    w = [2.0, -1.0, 0.5, 0.0]
    yl = [1.5 + sum(r[j] * w[j] for j in range(4)) + e
          for r, e in zip(Xl, eps)]
    pb = [1 / (1 + math.exp(-(0.5 + 1.5 * r[0] - r[1]))) for r in Xl]
    u = [float(v) for v in rng.uniform(0, 1, n)._flat()]
    yb = [1 if uu < p else 0 for uu, p in zip(u, pb)]
    return rng, Xl, yl, yb


def test_linear_and_ridge_exact():
    _, X, yl, _ = _data()
    m = A["ml"]
    g = ml.LinearRegression().fit(X, yl)
    assert g.intercept_ == pytest.approx(m["lin_intercept"], rel=1e-9)
    assert list(g.coef_._flat()) == pytest.approx(m["lin_coef"],
                                                  rel=1e-9)
    g = ml.Ridge(alpha=2.0).fit(X, yl)
    assert list(g.coef_._flat()) == pytest.approx(m["ridge_coef"],
                                                  rel=1e-8)
    gcv = ml.RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, yl)
    assert gcv.alpha_ == m["ridgecv_alpha"]


def test_logistic_same_l2_objective():
    _, X, _, yb = _data()
    m = A["ml"]
    g = ml.LogisticRegression(C=1.0).fit(X, yb)
    assert g.coef_.tolist()[0] == pytest.approx(m["logit_coef"],
                                                rel=5e-3, abs=5e-3)
    assert g.score(X, yb) == pytest.approx(m["logit_score"], abs=0.03)


def test_scaler_pca_isotonic_exact():
    rng, X, _, _ = _data()
    m = A["ml"]
    g = ml.StandardScaler().fit(X)
    assert g.mean_ == pytest.approx(m["scaler_mean"], rel=1e-12)
    assert g.scale_ == pytest.approx(m["scaler_scale"], rel=1e-12)
    gp = ml.PCA(n_components=2).fit(X)
    assert gp.explained_variance_ratio_.tolist() == pytest.approx(
        m["pca_evr"], rel=1e-9)
    xs = [float(v) for v in rng.uniform(0, 10, 60)._flat()]
    ys = [a * 0.5 + float(e) for a, e in
          zip(xs, rng.normal(0, 1, 60)._flat())]
    gi = ml.IsotonicRegression().fit(xs, ys)
    q = sorted(xs)
    assert gi.predict(q).tolist() == pytest.approx(m["iso_pred"],
                                                   rel=1e-9)


def test_forests_and_boosting_parity():
    _, X, yl, yb = _data()
    Xtr, Xte = X[:150], X[150:]
    ytr, yte = yl[:150], yl[150:]
    btr, bte = yb[:150], yb[150:]
    fw = A["ml_forest"]

    def r2(y_true, y_pred):
        ybar = sum(y_true) / len(y_true)
        ssr = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
        sst = sum((a - ybar) ** 2 for a in y_true)
        return 1.0 - ssr / sst

    g = ml.RandomForestRegressor(n_estimators=50, random_state=1
                                 ).fit(Xtr, ytr)
    r2g = r2(yte, _l(g.predict(Xte)))
    assert r2g > 0.7 and abs(r2g - fw["rf_r2"]) < 0.12
    g = ml.RandomForestClassifier(n_estimators=50, random_state=1
                                  ).fit(Xtr, btr)
    accg = sum(1 for a, b in zip(_l(g.predict(Xte)), bte)
               if int(a) == b) / len(bte)
    assert accg > 0.65 and abs(accg - fw["rfc_acc"]) < 0.12
    g = ml.GradientBoostingRegressor(n_estimators=60).fit(Xtr, ytr)
    r2g = r2(yte, _l(g.predict(Xte)))
    assert r2g > 0.8 and abs(r2g - fw["gb_r2"]) < 0.1
    g = ml.GradientBoostingClassifier(n_estimators=60).fit(Xtr, btr)
    accg = sum(1 for a, b in zip(_l(g.predict(Xte)), bte)
               if int(a) == b) / len(bte)
    assert accg > 0.65


def test_clustering_identical_on_blobs():
    from itertools import permutations
    rng = np.random.default_rng(23)
    B = np.vstack([rng.normal(0, .5, (40, 2)),
                   rng.normal(6, .5, (40, 2)),
                   rng.normal([0, 6], .5, (40, 2))])
    g = ml.KMeans(n_clusters=3, n_init=5, random_state=2).fit(B.tolist())
    lg = [int(v) for v in g.labels_.tolist()]
    lw = A["ml_cluster"]["km_labels"]
    match = max(sum(1 for a, b in zip(lg, lw) if p[a] == b)
                for p in permutations(range(3)))
    assert match == len(lg)
    gd = ml.DBSCAN(eps=1.2, min_samples=4).fit(B.tolist())
    ng = len(set(int(v) for v in gd.labels_.tolist() if v >= 0))
    assert ng == A["ml_cluster"]["db_nclusters"] == 3


def test_svm_parity_and_metrics():
    rng, X, _, yb = _data()
    Xtr, Xte = X[:150], X[150:]
    btr, bte = yb[:150], yb[150:]
    g = ml.SVC(C=1.0).fit(Xtr, btr)
    pred = [int(v) for v in _l(g.predict(Xte))]
    accg = sum(1 for a, b in zip(pred, bte) if a == b) / len(bte)
    assert accg > 0.6
    scores = [float(v) for v in rng.uniform(0, 1, len(yb))._flat()]
    assert ml.roc_auc_score(yb, scores) == pytest.approx(A["ml"]["auc"],
                                                         rel=1e-10)
    rep = ml.classification_report(bte, g.predict(Xte),
                                   output_dict=True)
    assert rep["accuracy"] == pytest.approx(accg)
