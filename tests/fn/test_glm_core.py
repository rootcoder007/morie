"""Equivalence tests: morie.fn._glm_core vs statsmodels."""

from morie.fn import _array_core as np
import pytest

sm_mod = pytest.importorskip(
    "statsmodels.api", reason="equivalence baseline needs statsmodels")
import statsmodels.api as sm
import statsmodels.formula.api as smf

from morie.fn import _glm_core as mg


def _data():
    rng = np.random.default_rng(17)
    n = 120
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(0, 0.8, n)
    X = sm.add_constant(np.column_stack([x1, x2]))
    return rng, x1, x2, y, X


def test_ols_full_inference():
    _, _, _, y, X = _data()
    g = mg.OLS(y.tolist(), X.tolist()).fit()
    w = sm.OLS(y, X).fit()
    assert list(g.params._flat()) == pytest.approx(list(w.params), rel=1e-9)
    assert list(g.bse._flat()) == pytest.approx(list(w.bse), rel=1e-9)
    assert g.rsquared == pytest.approx(w.rsquared, rel=1e-10)
    assert g.aic == pytest.approx(w.aic, rel=1e-8)
    assert g.fvalue == pytest.approx(w.fvalue, rel=1e-8)
    assert g.conf_int().tolist()[1] == pytest.approx(
        list(w.conf_int()[1]), rel=1e-8)


def test_ols_hc3_robust():
    _, _, _, y, X = _data()
    g = mg.OLS(y.tolist(), X.tolist()).fit(cov_type="HC3")
    w = sm.OLS(y, X).fit(cov_type="HC3")
    assert list(g.bse._flat()) == pytest.approx(list(w.bse), rel=1e-8)


def test_logit_matches():
    rng, x1, x2, y, X = _data()
    pb = 1 / (1 + np.exp(-(0.5 + 1.5 * x1 - x2)))
    yb = (rng.uniform(0, 1, len(x1)) < pb).astype(float)
    g = mg.Logit(yb.tolist(), X.tolist()).fit()
    w = sm.Logit(yb, X).fit(disp=0)
    assert list(g.params._flat()) == pytest.approx(list(w.params),
                                                   rel=1e-6)
    assert g.llf == pytest.approx(w.llf, rel=1e-8)


def test_glm_poisson_matches():
    rng, x1, _, _, _ = _data()
    mu = np.exp(0.3 + 0.7 * x1)
    yp = rng.poisson(mu).astype(float)
    Xp = sm.add_constant(x1)
    g = mg.GLM(yp.tolist(), Xp.tolist(), family=mg.Poisson()).fit()
    w = sm.GLM(yp, Xp, family=sm.families.Poisson()).fit()
    assert list(g.params._flat()) == pytest.approx(list(w.params),
                                                   rel=1e-8)
    assert g.deviance == pytest.approx(w.deviance, rel=1e-7)


def test_wls_matches():
    rng, _, _, y, X = _data()
    wts = rng.uniform(0.5, 2.0, len(y))
    g = mg.WLS(y.tolist(), X.tolist(), weights=wts.tolist()).fit()
    w = sm.WLS(y, X, weights=wts).fit()
    assert list(g.params._flat()) == pytest.approx(list(w.params),
                                                   rel=1e-9)
    assert list(g.bse._flat()) == pytest.approx(list(w.bse), rel=1e-9)


def test_formula_api_with_categorical_and_interaction():
    import pandas as rp
    rng, x1, x2, y, _ = _data()
    gcat = rng.choice(["a", "b", "c"], len(y))
    native = {"y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist(),
              "g": gcat.tolist()}
    df = rp.DataFrame(native)
    g = mg.ols("y ~ x1 + x2 + C(g) + x1:x2", native).fit()
    w = smf.ols("y ~ x1 + x2 + C(g) + x1:x2", df).fit()
    for nm in ("Intercept", "x1", "C(g)[T.b]", "C(g)[T.c]", "x1:x2"):
        assert g.params[nm] == pytest.approx(w.params[nm], rel=1e-8)


def test_add_constant_shapes():
    out = mg.add_constant([[1.0, 2.0], [3.0, 4.0]])
    assert out.tolist() == [[1.0, 1.0, 2.0], [1.0, 3.0, 4.0]]
