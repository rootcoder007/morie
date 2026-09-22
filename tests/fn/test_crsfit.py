"""Tests for crsfit.cross_fit_one_step."""

from morie.fn import _array_core as np

from morie.fn.crsfit import cross_fit_one_step


def _nuis(y_tr, d_tr, X_tr, X_te):
    """OLS/logit nuisance for the AIPW score."""
    Xt = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
    Xe = np.hstack([np.ones((X_te.shape[0], 1)), X_te])
    # OLS on treated / control separately
    Xt1 = Xt[d_tr == 1.0]
    yt1 = y_tr[d_tr == 1.0]
    Xt0 = Xt[d_tr == 0.0]
    yt0 = y_tr[d_tr == 0.0]
    b1 = np.linalg.lstsq(Xt1, yt1, rcond=None)[0]
    b0 = np.linalg.lstsq(Xt0, yt0, rcond=None)[0]
    m1 = Xe @ b1
    m0 = Xe @ b0
    # logistic propensity via gradient descent on log-loss
    B = Xt
    w = np.zeros(B.shape[1])
    for _ in range(200):
        z = B @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = B.T @ (p - d_tr) / B.shape[0]
        w = w - 1.0 * grad
    e = 1.0 / (1.0 + np.exp(-(Xe @ w)))
    return m1, m0, e


def test_crsfit_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 2))
    d = (rng.uniform(size=n) < 0.5).astype(float)
    y = 2.0 * d + X[:, 0] + rng.normal(size=n)

    result = cross_fit_one_step(y, d, X, _nuis, n_folds=5, seed=0, trunc=0.01)
    assert isinstance(result, dict)
    assert "estimate" in result

    # Independent numeric expectation: AIPW score with full-sample nuisances
    # gives the no-cross-fit benchmark the function also reports.
    m1f, m0f, ef = _nuis(y, d, X, X)
    trunc = 0.01
    ef = np.clip(ef, trunc, 1 - trunc)
    expected_naive = float(np.mean(
        m1f - m0f + d * (y - m1f) / ef - (1 - d) * (y - m0f) / (1 - ef)
    ))
    assert abs(result["no_crossfit_estimate"] - expected_naive) < 1e-8

    # And the cross-fit estimate should be reasonably close to the true
    # treatment effect of 2.0.
    assert abs(result["estimate"] - 2.0) < 0.6


def test_crsfit_edge():
    """Test edge cases."""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 2))
    d = (rng.uniform(size=n) < 0.5).astype(float)
    y = 2.0 * d + X[:, 0] + rng.normal(size=n)

    result = cross_fit_one_step(y, d, X, _nuis, n_folds=5, seed=0, trunc=0.01)
    assert isinstance(result, dict)
    assert "no_crossfit_estimate" in result
    assert "own_observation_bias" in result
