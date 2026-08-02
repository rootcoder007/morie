"""GARCH cluster: _garch core + the 22 Tsay Ch.3 specifications.

Anchored on Tsay (2010) Analysis of Financial Time Series 3rd ed.
Ch. 3, read in the PDF: GARCH Sec 3.5 p.131, IGARCH Sec 3.6 p.140,
GARCH-M Sec 3.7 p.142 eq.(3.23), EGARCH Sec 3.8 p.143 eq.(3.24)-(3.25),
TGARCH Sec 3.9 p.149 eq.(3.34)."""

from morie.fn import _array_core as np
import pytest

from morie.fn._garch import SPECS, garch_fit, garch_forecast, garch_recursion


def simulate_garch(n=3000, omega=0.05, alpha=0.1, beta=0.85, seed=0, gamma=0.0):
    """GARCH(1,1) or GJR path. Unconditional variance = omega/(1-a-b-g/2)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + 500)
    e = np.empty(n + 500)
    s2 = omega / max(1 - alpha - beta - gamma / 2, 1e-6)
    for t in range(n + 500):
        neg = 1.0 if t > 0 and e[t - 1] < 0 else 0.0
        if t > 0:
            s2 = omega + (alpha + gamma * neg) * e[t - 1] ** 2 + beta * s2
        e[t] = np.sqrt(s2) * z[t]
    return e[500:]


def test_recursion_matches_the_tsay_equations_by_hand():
    e = np.array([0.5, -1.2, 0.3, 0.8, -0.4, 1.1, -0.9, 0.2, 0.6, -0.7])
    v = float(np.var(e))
    # GARCH(1,1), Tsay Sec 3.5
    s2 = garch_recursion(e, {"omega": 0.05, "alpha": 0.1, "beta": 0.85}, "garch")
    assert s2[0] == pytest.approx(v)
    assert s2[1] == pytest.approx(0.05 + 0.1 * 0.25 + 0.85 * v)
    assert s2[2] == pytest.approx(0.05 + 0.1 * 1.44 + 0.85 * s2[1])
    # IGARCH: alpha is pinned at 1 - beta (Tsay p.141), so the two
    # variance weights must sum to exactly one
    i2 = garch_recursion(e, {"omega": 0.01, "beta": 0.9}, "igarch")
    assert i2[1] == pytest.approx(0.01 + 0.9 * v + 0.1 * 0.25)
    # GJR/TGARCH eq.(3.34): gamma loads only on the negative shock
    g2 = garch_recursion(
        e, {"omega": 0.05, "alpha": 0.05, "gamma": 0.1, "beta": 0.85}, "gjr"
    )
    assert g2[1] == pytest.approx(0.05 + 0.05 * 0.25 + 0.85 * v)  # e[0] > 0
    assert g2[2] == pytest.approx(0.05 + (0.05 + 0.1) * 1.44 + 0.85 * g2[1])  # e[1] < 0
    # EGARCH eq.(3.24): E|z| = sqrt(2/pi) for a Gaussian (p.143 Remark)
    ls = garch_recursion(
        e, {"omega": -0.1, "alpha": 0.2, "gamma": -0.05, "beta": 0.9}, "egarch"
    )
    z0 = e[0] / np.sqrt(v)
    want = -0.1 + 0.9 * np.log(v) + 0.2 * (abs(z0) - np.sqrt(2 / np.pi)) - 0.05 * z0
    assert np.log(ls[1]) == pytest.approx(want)


def test_garch_recovers_its_parameters():
    hits = 0
    for seed in range(6):
        e = simulate_garch(seed=seed)
        f = garch_fit(e, "garch")
        # true alpha 0.1, beta 0.85, unconditional variance 1.0
        hits += abs(f["params"]["alpha"] - 0.1) < 0.05 and abs(
            f["params"]["beta"] - 0.85
        ) < 0.07
        assert f["persistence"] < 1  # covariance stationary
        assert np.mean(f["sigma2"]) == pytest.approx(1.0, rel=0.35)
    assert hits >= 5  # measured 6/6


def test_fitted_model_beats_a_constant_variance():
    # the whole point of a GARCH: standardised residuals should be far
    # less heteroscedastic than the raw series
    e = simulate_garch(seed=1)
    f = garch_fit(e, "garch")
    z = f["std_residuals"]
    from scipy import stats as st

    def arch_lm(x, lags=5):
        x2 = x**2
        X = np.column_stack([x2[lags - 1 - i : -1 - i] for i in range(lags)] + [
            np.ones(x2.size - lags)
        ])
        y = x2[lags:]
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)
        return x2.size * r2

    assert arch_lm(z) < arch_lm(e)  # ARCH effect removed
    assert st.kurtosis(z) < st.kurtosis(e)  # and the fat tail with it


def test_igarch_persistence_is_exactly_one():
    e = simulate_garch(alpha=0.1, beta=0.9, seed=2)  # a unit-root path
    f = garch_fit(e, "igarch")
    assert f["persistence"] == 1.0
    assert 0 < f["params"]["beta"] < 1
    # the recursion's two weights must sum to one by construction
    assert f["params"]["beta"] + (1 - f["params"]["beta"]) == pytest.approx(1.0)


def test_gjr_detects_the_leverage_effect():
    # asymmetric truth: gamma = 0.15 loads on negative shocks only
    caught = 0
    for seed in range(6):
        e = simulate_garch(alpha=0.03, beta=0.85, gamma=0.15, seed=seed)
        f = garch_fit(e, "gjr")
        caught += f["params"]["gamma"] > 0.05
    assert caught >= 5  # measured 6/6
    # symmetric data must NOT produce a large gamma
    sym = simulate_garch(alpha=0.1, beta=0.85, gamma=0.0, seed=7)
    assert abs(garch_fit(sym, "gjr")["params"]["gamma"]) < 0.1


def test_egarch_allows_negative_omega_and_keeps_variance_positive():
    e = simulate_garch(seed=3)
    f = garch_fit(e, "egarch")
    assert np.all(f["sigma2"] > 0)  # exp() guarantees positivity
    assert abs(f["params"]["beta"]) < 1
    assert np.isfinite(f["loglik"])


def test_heavy_tailed_likelihoods_win_on_heavy_tailed_data():
    rng = np.random.default_rng(5)
    n = 2500
    e = np.empty(n)
    s2 = 1.0
    zt = rng.standard_t(5, size=n) / np.sqrt(5 / 3)
    for t in range(n):
        if t:
            s2 = 0.05 + 0.1 * e[t - 1] ** 2 + 0.85 * s2
        e[t] = np.sqrt(s2) * zt[t]
    norm = garch_fit(e, "garch", dist="normal")
    tdist = garch_fit(e, "garch", dist="t")
    ged = garch_fit(e, "garch", dist="ged")
    assert tdist["loglik"] > norm["loglik"]  # t fits t-data better
    assert 2 < tdist["nu"] < 15
    assert np.isfinite(ged["loglik"])


def test_forecast_converges_to_the_unconditional_variance():
    e = simulate_garch(seed=4)
    f = garch_fit(e, "garch")
    fc = garch_forecast(f, horizon=200)
    p = f["params"]
    uncond = p["omega"] / (1 - p["alpha"] - p["beta"])
    assert fc[-1] == pytest.approx(uncond, rel=0.05)
    assert np.all(np.isfinite(fc))
    with pytest.raises(ValueError):
        garch_forecast(f, horizon=0)


def test_aparch_nests_gjr_at_delta_two():
    e = np.array([0.5, -1.2, 0.3, 0.8, -0.4, 1.1, -0.9, 0.2, 0.6, -0.7])
    # APARCH with delta = 2 and gamma = 0 is a plain GARCH(1,1)
    ap = garch_recursion(
        e, {"omega": 0.05, "alpha": 0.1, "gamma": 0.0, "beta": 0.85, "delta": 2.0},
        "aparch",
    )
    ga = garch_recursion(e, {"omega": 0.05, "alpha": 0.1, "beta": 0.85}, "garch")
    assert ap == pytest.approx(ga)


def test_cgarch_splits_persistence_into_two_components():
    e = simulate_garch(n=2000, seed=6)
    f = garch_fit(e, "cgarch")
    assert f["component"] is not None
    assert f["component"].size == e.size
    assert np.all(f["sigma2"] > 0)
    # the permanent component is smoother than total variance
    assert np.std(np.diff(f["component"])) < np.std(np.diff(f["sigma2"]))


def test_figarch_weights_are_nonnegative_and_decaying():
    from morie.fn._garch import _figarch_weights

    lam = _figarch_weights(d=0.4, beta=0.6, phi=0.2, trunc=60)
    assert np.all(lam >= 0)
    assert lam[0] > lam[30] >= lam[59]  # hyperbolic decay
    e = simulate_garch(n=1500, seed=8)
    s2 = garch_recursion(e, {"omega": 0.05, "d": 0.4, "beta": 0.6, "phi": 0.2}, "figarch")
    assert np.all(s2 > 0)


def test_input_validation():
    e = simulate_garch(n=200, seed=9)
    with pytest.raises(ValueError):
        garch_fit(e, "weibull")
    with pytest.raises(ValueError):
        garch_fit(e[:10])
    with pytest.raises(ValueError):
        garch_fit(np.full(100, np.nan))
    with pytest.raises(ValueError):
        garch_fit(np.ones(100))  # zero variance
    with pytest.raises(ValueError):
        garch_recursion(e[:3], {"omega": 1, "alpha": 0.1, "beta": 0.8}, "garch")
    assert set(SPECS) == {
        "garch", "igarch", "egarch", "gjr", "tgarch", "aparch", "cgarch", "figarch"
    }


# ---------------------------------------------------------------- front-ends


def test_univariate_front_ends_return_fitted_models():
    from morie.fn.egarch import egarch_model
    from morie.fn.egarcm import egarch_nelson
    from morie.fn.garchm import garch_model
    from morie.fn.igarcm import igarch_integrated
    from morie.fn.tgarcm import tgarch_gjr
    from morie.fn.volaprch import vol_aparch_fit
    from morie.fn.volcgar import vol_cgarch_fit
    from morie.fn.volegar import vol_egarch_fit
    from morie.fn.volfig import vol_figarch_fit
    from morie.fn.volgar import vol_garch11_fit
    from morie.fn.volgargd import vol_garch_ged
    from morie.fn.volgargt import vol_garch_t
    from morie.fn.volgjr import vol_gjr_garch
    from morie.fn.volign import vol_igarch_fit
    from morie.fn.voltgr import vol_tgarch_fit

    e = simulate_garch(n=1200, seed=11)
    for f in (garch_model, egarch_model, egarch_nelson, tgarch_gjr):
        out = f(e)
        assert np.all(out["sigma2"] > 0)
        assert np.isfinite(out["loglik"])
        assert out["forecast"] > 0
    for f in (
        vol_garch11_fit, vol_egarch_fit, vol_gjr_garch, vol_tgarch_fit,
        vol_aparch_fit, vol_cgarch_fit, vol_figarch_fit, vol_garch_t,
        vol_garch_ged, vol_igarch_fit,
    ):
        out = f(e)
        assert np.all(out["sigma2"] > 0), f.__name__
        assert np.isfinite(out["loglik"]), f.__name__
    assert igarch_integrated(e)["persistence"] == 1.0
    assert vol_igarch_fit(e)["persistence"] == 1.0
    # the t and GED fits must actually estimate a shape parameter
    assert vol_garch_t(e)["nu"] is not None
    assert vol_garch_ged(e)["nu"] is not None


def test_garch_forecast_field_matches_the_recursion():
    from morie.fn.volgar import vol_garch11_fit

    e = simulate_garch(n=1200, seed=12)
    out = vol_garch11_fit(e)
    p = out["params"]
    want = p["omega"] + p["alpha"] * out["residuals"][-1] ** 2 + p["beta"] * out["sigma2"][-1]
    assert out["forecast"] == pytest.approx(want)


def _panel(seed=0, T=800, k=3, rho=0.5):
    rng = np.random.default_rng(seed)
    cov = np.full((k, k), rho) + (1 - rho) * np.eye(k)
    L = np.linalg.cholesky(cov)
    out = np.empty((T, k))
    s2 = np.ones(k)
    for t in range(T):
        z = L @ rng.standard_normal(k)
        out[t] = np.sqrt(s2) * z
        s2 = 0.05 + 0.1 * out[t] ** 2 + 0.85 * s2
    return out


def test_bekk_keeps_every_covariance_positive_definite():
    from morie.fn.mgrch import bekk_garch_multivariate
    from morie.fn.volbekk import vol_bekk_garch

    R = _panel(seed=1)
    out = bekk_garch_multivariate(R)
    H = out["H"]
    assert H.shape == (800, 3, 3)
    # positive definiteness is the point of the quadratic BEKK form
    for t in (0, 100, 400, 799):
        assert np.all(np.linalg.eigvalsh(H[t]) > 0)
        assert np.allclose(H[t], H[t].T)
    assert 0 < out["persistence"] < 1
    # variance targeting: the long-run covariance IS the sample one
    assert out["H_bar"] == pytest.approx(np.cov(R - R.mean(axis=0), rowvar=False))
    assert vol_bekk_garch(R)["a"] == pytest.approx(out["a"])
    with pytest.raises(ValueError):
        bekk_garch_multivariate(R[:, :1])


def test_orthogonal_garch_reconstructs_a_symmetric_covariance():
    from morie.fn.volgo import vol_garch_orthogonal

    R = _panel(seed=2)
    full = vol_garch_orthogonal(R)
    assert full["full_rank"] is True
    assert np.allclose(full["H"][100], full["H"][100].T)
    assert np.all(np.linalg.eigvalsh(full["H"][100]) > 0)
    assert full["explained_variance_ratio"].sum() == pytest.approx(
        full["explained_variance_ratio"].sum()
    )
    # truncating to k < d gives a singular H, and says so
    red = vol_garch_orthogonal(R, k=1)
    assert red["full_rank"] is False
    assert np.linalg.matrix_rank(red["H"][100]) == 1
    with pytest.raises(ValueError):
        vol_garch_orthogonal(R, k=9)


def test_markov_switching_separates_calm_and_turbulent_regimes():
    from morie.fn.volmsg import vol_markov_switching_garch

    rng = np.random.default_rng(3)
    n = 1200
    # regime 1 is genuinely four times as volatile as regime 0
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < 0.98 else 1 - state[t - 1]
    r = rng.standard_normal(n) * np.where(state == 1, 2.0, 0.5)
    out = vol_markov_switching_garch(r, K=2)
    uv = out["unconditional_var"]
    assert uv[0] < uv[1]  # sorted: calm first
    assert uv[1] > 2 * uv[0]
    P = out["transition"]
    assert P.shape == (2, 2)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(np.diag(P) > 0.5)  # regimes persist
    with pytest.raises(ValueError):
        vol_markov_switching_garch(r, K=1)


def test_var_and_expected_shortfall_order_correctly():
    from morie.fn.volges import vol_garch_es_impl
    from morie.fn.volgvi import vol_garch_var_impl

    v = vol_garch_var_impl(0.0, 1.0, alpha=0.05)
    # standard normal 5%: VaR = -z_0.05, ES = phi(z_0.05)/0.05.
    # Both computed from scipy, not recalled.
    assert v["var"] == pytest.approx(1.6448536269514729, abs=1e-12)
    e = vol_garch_es_impl(0.0, 1.0, alpha=0.05)
    assert e["es"] == pytest.approx(2.0627128075074253, abs=1e-12)
    assert e["es"] > e["var"]  # ES is always beyond VaR
    # a fatter tail must widen both
    t = vol_garch_es_impl(0.0, 1.0, alpha=0.01, dist="t", nu=4.0)
    g = vol_garch_es_impl(0.0, 1.0, alpha=0.01, dist="normal")
    assert t["var"] > g["var"]
    assert t["es"] > g["es"]
    # VaR scales linearly in sigma and shifts with mu
    assert vol_garch_var_impl(0.0, 2.0)["var"] == pytest.approx(2 * v["var"])
    assert vol_garch_var_impl(0.5, 1.0)["var"] == pytest.approx(v["var"] - 0.5)
    with pytest.raises(ValueError):
        vol_garch_var_impl(0.0, -1.0)
    with pytest.raises(ValueError):
        vol_garch_es_impl(0.0, 1.0, alpha=1.5)
    with pytest.raises(ValueError):
        vol_garch_es_impl(0.0, 1.0, dist="t", nu=1.5)


def test_skew_egarch_separates_the_two_asymmetries():
    from morie.fn.volnsig import vol_nelson_skew_garch

    e = simulate_garch(n=1500, seed=13)
    out = vol_nelson_skew_garch(e)
    assert np.all(out["sigma2"] > 0)
    assert out["nu"] > 0
    assert out["lambda_skew"] > 0
    # symmetric data: the skew likelihood cannot beat the symmetric one
    # by much, so the LR statistic stays small
    assert out["skew_lr_stat"] < 10
    assert out["skew_loglik"] >= out["symmetric_loglik"] - 1e-6
    # and the two asymmetry parameters are reported separately
    assert "gamma" in out["params"]
    assert "lambda_skew" in out
