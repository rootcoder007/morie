"""Tests for volccc.vol_ccc_garch (Bollerslev 1990).

Properties come from the model definition H_t = D_t R D_t, equation (5)
of the rmgarch model reference, plus the closed-form second step.
"""

import numpy as np
import pytest

from morie.fn.volccc import vol_ccc_garch


def _panel(seed=3, n=400, rho=0.7):
    """Two GARCH-like series with a known constant correlation."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
    h = np.empty((n, 2))
    r = np.empty((n, 2))
    for j in range(2):
        h[0, j] = 1.0
        r[0, j] = np.sqrt(h[0, j]) * z[0, j]
        for t in range(1, n):
            h[t, j] = 0.05 + 0.10 * r[t - 1, j] ** 2 + 0.85 * h[t - 1, j]
            r[t, j] = np.sqrt(h[t, j]) * z[t, j]
    return r


def test_R_is_a_valid_correlation_matrix():
    res = vol_ccc_garch(_panel())
    R = res["R"]
    assert R.shape == (2, 2)
    assert np.allclose(np.diag(R), 1.0)
    assert np.allclose(R, R.T)
    assert np.all(np.linalg.eigvalsh(R) > 0), "R must be positive definite"


def test_recovers_the_constant_correlation():
    """R is the correlation of the standardised residuals, so it should
    land near the rho the data were generated with."""
    res = vol_ccc_garch(_panel(rho=0.7))
    assert res["R"][0, 1] == pytest.approx(0.7, abs=0.10)


def test_zero_correlation_recovered():
    """Averaged over seeds, not asserted on one draw.

    At n=400 a single sample correlation has a standard error near 0.05,
    so an individual seed can sit 0.11 away from the truth without any
    defect in the estimator. What must hold is that the estimate is
    centred on zero.
    """
    est = [vol_ccc_garch(_panel(seed=s, rho=0.0))["R"][0, 1] for s in range(8)]
    assert float(np.mean(est)) == pytest.approx(0.0, abs=0.04)


def test_conditional_variance_is_positive_and_shaped():
    res = vol_ccc_garch(_panel())
    H = res["conditional_variance"]
    assert H.shape == (400, 2)
    assert np.all(H > 0)
    assert np.allclose(res["sigmas"], np.sqrt(H))


def test_loglik_is_finite():
    res = vol_ccc_garch(_panel())
    assert np.isfinite(res["ll"])
    assert res["ll"] == res["loglik"]


def test_transposed_panel_is_detected():
    X = _panel()
    assert vol_ccc_garch(X.T)["k"] == 2


def test_rejects_single_series():
    with pytest.raises(ValueError, match="k>=2"):
        vol_ccc_garch(np.random.default_rng(0).normal(0, 1, 100))


def test_rejects_short_sample():
    with pytest.raises(ValueError, match="n>=30"):
        vol_ccc_garch(np.random.default_rng(0).normal(0, 1, (20, 2)))
