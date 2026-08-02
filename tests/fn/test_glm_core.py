"""Equivalence tests: morie.fn._glm_core vs frozen statsmodels anchors.

The reference values in oracle_anchors.json were computed ONCE from
statsmodels 0.14.6 / numpy 2.4.4 on exactly the inputs these tests
regenerate from morie's own RNG (Philox, seed 17). The library itself
is not imported anywhere; the anchors file records the versions.
"""
import json
import math
import pathlib

import pytest

from morie.fn import _array_core as np
from morie.fn import _glm_core as mg

A = json.loads(pathlib.Path(__file__).with_name(
    "oracle_anchors.json").read_text())


def _data():
    rng = np.random.default_rng(17)
    n = 120
    x1 = [float(v) for v in rng.normal(0, 1, n)._flat()]
    x2 = [float(v) for v in rng.normal(0, 1, n)._flat()]
    eps = [float(v) for v in rng.normal(0, 0.8, n)._flat()]
    y = [1.0 + 2.0 * a - 0.5 * b + e for a, b, e in zip(x1, x2, eps)]
    X = [[1.0, a, b] for a, b in zip(x1, x2)]
    return rng, x1, x2, y, X


def test_ols_full_inference():
    _, _, _, y, X = _data()
    g = mg.OLS(y, X).fit()
    w = A["ols"]
    assert list(g.params._flat()) == pytest.approx(w["params"], rel=1e-9)
    assert list(g.bse._flat()) == pytest.approx(w["bse"], rel=1e-9)
    assert g.rsquared == pytest.approx(w["rsquared"], rel=1e-10)
    assert g.aic == pytest.approx(w["aic"], rel=1e-8)
    assert g.fvalue == pytest.approx(w["fvalue"], rel=1e-8)
    ci = g.conf_int()
    row1 = ci.values.tolist()[1] if hasattr(ci, "values") \
        else ci.tolist()[1]
    assert row1 == pytest.approx(w["conf_int_row1"], rel=1e-8)


def test_ols_hc3_robust():
    _, _, _, y, X = _data()
    g = mg.OLS(y, X).fit(cov_type="HC3")
    assert list(g.bse._flat()) == pytest.approx(A["ols_hc3_bse"],
                                                rel=1e-8)


def test_logit_matches():
    rng, x1, x2, y, X = _data()
    pb = [1 / (1 + math.exp(-(0.5 + 1.5 * a - b)))
          for a, b in zip(x1, x2)]
    u = [float(v) for v in rng.uniform(0, 1, len(x1))._flat()]
    yb = [1.0 if uu < p else 0.0 for uu, p in zip(u, pb)]
    g = mg.Logit(yb, X).fit()
    assert list(g.params._flat()) == pytest.approx(A["logit"]["params"],
                                                   rel=1e-6)
    assert g.llf == pytest.approx(A["logit"]["llf"], rel=1e-8)


def test_glm_poisson_matches():
    rng, x1, _, _, _ = _data()
    mu = [math.exp(0.3 + 0.7 * a) for a in x1]
    yp = rng.poisson(np.marr(mu))
    Xp = [[1.0, a] for a in x1]
    g = mg.GLM([float(v) for v in yp._flat()], Xp,
               family=mg.Poisson()).fit()
    assert list(g.params._flat()) == pytest.approx(
        A["poisson"]["params"], rel=1e-8)
    assert g.deviance == pytest.approx(A["poisson"]["deviance"],
                                       rel=1e-7)


def test_wls_matches():
    rng, _, _, y, X = _data()
    wts = [float(v) for v in rng.uniform(0.5, 2.0, len(y))._flat()]
    g = mg.WLS(y, X, weights=wts).fit()
    assert list(g.params._flat()) == pytest.approx(A["wls"]["params"],
                                                   rel=1e-9)
    assert list(g.bse._flat()) == pytest.approx(A["wls"]["bse"],
                                                rel=1e-9)


def test_formula_api_with_categorical_and_interaction():
    rng, x1, x2, y, _ = _data()
    gcat = [str(v) for v in rng.choice(np.oarr(["a", "b", "c"]),
                                       len(y))]
    native = {"y": y, "x1": x1, "x2": x2, "g": gcat}
    g = mg.ols("y ~ x1 + x2 + C(g) + x1:x2", native).fit()
    for nm in ("Intercept", "x1", "C(g)[T.b]", "C(g)[T.c]", "x1:x2"):
        assert g.params[nm] == pytest.approx(A["formula"][nm], rel=1e-8)


def test_add_constant_shapes():
    out = mg.add_constant([[1.0, 2.0], [3.0, 4.0]])
    assert out.tolist() == [[1.0, 1.0, 2.0], [1.0, 3.0, 4.0]]
