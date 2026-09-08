"""Lewbel's special-regressor estimator (Horowitz Sec. 4.5)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.hrzlew import horowitz_lewbel_estimator


def _draw(n, rng, beta=np.array([0.0, 1.0]), hetero=True):
    """Y = I(V + X'beta + eps > 0) with V additive and coefficient 1.

    V needs large support relative to X'beta + eps, so it is drawn
    with a wider scale than the index it has to cover.
    """
    x1 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1])
    V = rng.standard_normal(n) * 6.0
    scale = (1.0 + 0.8 * np.abs(x1)) if hetero else np.ones(n)
    eps = rng.standard_normal(n) * scale
    y = (V + X @ beta + eps > 0).astype(float)
    return X, y, V


def test_lewbel_recovers_beta_under_unknown_heteroskedasticity():
    rng = np.random.default_rng(0)
    beta = np.array([0.0, 1.0])
    est = []
    for s in range(6):
        X, y, V = _draw(6000, np.random.default_rng(s), beta)
        est.append(horowitz_lewbel_estimator(X, y, V)["beta"][1])
    # the error scale depends on x1, which is exactly the case probit
    # gets wrong and this estimator is built for
    assert abs(np.median(est) - beta[1]) < 0.25
    out = horowitz_lewbel_estimator(*_draw(4000, rng, beta))
    assert out["root_n_consistent"] is True
    assert out["heteroskedasticity_allowed"] is True
    assert out["coefficient_on_V"] == 1.0
    assert out["endogenous"] is False
    assert out["se"].shape == out["beta"].shape


def test_lewbel_converges_as_n_grows():
    def err(n):
        return np.median([
            abs(horowitz_lewbel_estimator(*_draw(n, np.random.default_rng(s)))["beta"][1] - 1.0)
            for s in range(6)])
    assert err(8000) < err(500)


def test_the_normal_shortcut_agrees_with_the_kernel_density():
    rng = np.random.default_rng(3)
    X, y, V = _draw(4000, rng)
    a = horowitz_lewbel_estimator(X, y, V)
    b = horowitz_lewbel_estimator(X, y, V, density="normal")
    # U really is normal here, so Estimator 1's parametric shortcut
    # and the kernel density must land in the same place
    assert abs(a["beta"][1] - b["beta"][1]) < 0.15
    assert b["bandwidth"] is None
    assert a["bandwidth"] > 0


def test_the_indicator_direction_matters():
    # T = [Y - I(V >= 0)] / f(U). Flipping the indicator to I(V < 0)
    # -- as at least one secondary description of this estimator
    # states -- does NOT give the same estimand.
    rng = np.random.default_rng(1)
    X, y, V = _draw(6000, rng)
    out = horowitz_lewbel_estimator(X, y, V)
    from morie.fn._horowitz import kernel, silverman_bw
    Vc = V - V.mean()
    coef, *_ = np.linalg.lstsq(X, Vc, rcond=None)
    U = Vc - X @ coef
    h = silverman_bw(U)
    f = kernel((U[:, None] - U[None, :]) / h).sum(axis=1) / (U.size * h)
    flipped, *_ = np.linalg.lstsq(X, (y - (V < 0)) / f, rcond=None)
    assert abs(out["beta"][1] - 1.0) < abs(flipped[1] - 1.0)


def test_lewbel_reports_the_weight_it_is_placing_on_the_tails():
    rng = np.random.default_rng(4)
    X, y, V = _draw(2000, rng)
    out = horowitz_lewbel_estimator(X, y, V)
    # 1/f(U) is a genuine weight and it is large in the tails: this is
    # the estimator's known fragility, reported rather than hidden
    assert out["max_weight"] == pytest.approx(1.0 / out["min_density"])
    assert out["max_weight"] > 1.0


def test_lewbel_validates_its_inputs():
    rng = np.random.default_rng(5)
    X, y, V = _draw(500, rng)
    with pytest.raises(ValueError):
        horowitz_lewbel_estimator(X, y, V[:100])
    with pytest.raises(ValueError):
        horowitz_lewbel_estimator(X, y * 2, V)
    with pytest.raises(ValueError):
        horowitz_lewbel_estimator(X, y, V, density="cauchy")
    with pytest.raises(ValueError):
        horowitz_lewbel_estimator(X, y, V, bandwidth=-1.0)
    with pytest.raises(ValueError):
        horowitz_lewbel_estimator(X[:5], y[:5], V[:5])


def test_lewbel_with_instruments_runs_two_stage_least_squares():
    rng = np.random.default_rng(7)
    n = 6000
    zi = rng.standard_normal(n)
    u = rng.standard_normal(n)
    x1 = zi + u  # endogenous: correlated with the error through u
    X = np.column_stack([np.ones(n), x1])
    V = rng.standard_normal(n) * 6.0
    eps = u + rng.standard_normal(n)
    y = (V + X @ np.array([0.0, 1.0]) + eps > 0).astype(float)
    ols = horowitz_lewbel_estimator(X, y, V)
    tsls = horowitz_lewbel_estimator(X, y, V, instruments=zi)
    assert tsls["endogenous"] is True
    assert abs(tsls["beta"][1] - 1.0) < abs(ols["beta"][1] - 1.0)
