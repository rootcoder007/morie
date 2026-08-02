"""Tests for whtnse.white_noise_test (Hosking 1980).

The generated placeholders asserted only that a dict came back. They are
replaced by checks of what the statistic must do: hold its size on white
noise, detect serial dependence, and satisfy the algebraic identities in
Hosking's definition.
"""

from morie.fn import _array_core as np
import pytest
from morie.fn import _stats_core as stats

from morie.fn.whtnse import _autocov, white_noise_test


def _white(n=400, k=3, seed=0):
    return np.random.default_rng(seed).normal(0, 1, (n, k))


def _var1(n=400, k=3, seed=0, a=0.6):
    """A VAR(1): every column depends on its own previous value."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, (n, k))
    x = np.empty((n, k))
    x[0] = e[0]
    for t in range(1, n):
        x[t] = a * x[t - 1] + e[t]
    return x


def test_white_noise_is_not_rejected():
    assert white_noise_test(_white(seed=1), lags=10)["p_value"] > 0.05


def test_serially_dependent_series_is_rejected():
    assert white_noise_test(_var1(seed=2), lags=10)["p_value"] < 0.01


def test_degrees_of_freedom_follow_k_squared_times_lags():
    assert white_noise_test(_white(k=3, seed=3), lags=7)["df"] == 9 * 7


def test_fitdf_reduces_the_degrees_of_freedom():
    assert white_noise_test(_white(k=2, seed=4), lags=10, fitdf=3)["df"] == 4 * (10 - 3)


def test_statistic_is_non_negative():
    """Each term is tr(A' S A S) with S positive definite, so none is negative."""
    for seed in range(5):
        assert white_noise_test(_white(seed=seed), lags=8)["statistic"] >= 0


def test_statistic_grows_with_more_lags():
    """The sum is over non-negative terms, so it cannot fall as m rises."""
    X = _white(seed=6)
    vals = [white_noise_test(X, lags=m)["statistic"] for m in (2, 5, 10, 20)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_modified_weighting_exceeds_the_unmodified_one():
    """n^2/(n - l) > n for every l >= 1, so the modified form is larger."""
    X = _white(seed=7)
    a = white_noise_test(X, lags=10, modified=True)["statistic"]
    b = white_noise_test(X, lags=10, modified=False)["statistic"]
    assert a > b


def test_matches_the_univariate_scalar_case():
    """With k = 1 the trace term is the squared autocorrelation.

    Gamma_l' G0^-1 Gamma_l G0^-1 collapses to (gamma_l / gamma_0)^2, so
    the statistic reduces to n^2 sum r_l^2 / (n - l), the Ljung-Box form.
    This pins the algebra without a reference implementation.
    """
    x = np.random.default_rng(8).normal(0, 1, (300, 1))
    m = 6
    res = white_noise_test(x, lags=m)
    xc = x - x.mean()
    g0 = ((xc.T @ xc) / 300).item()
    expected = 300**2 * sum(
        (((xc[l:].T @ xc[:-l]) / 300).item() / g0) ** 2 / (300 - l) for l in range(1, m + 1)
    )
    assert res["statistic"] == pytest.approx(expected, rel=1e-12)


def test_autocov_matches_its_definition():
    E = _white(n=50, k=2, seed=9)
    E = E - E.mean(axis=0)
    manual = sum(np.outer(E[t], E[t - 3]) for t in range(3, 50)) / 50
    assert np.allclose(_autocov(E, 3), manual)


def test_supplied_cdf_replaces_the_asymptotic_null():
    X = _white(seed=10)
    res = white_noise_test(X, lags=5, cdf=stats.chi2(9 * 5).cdf)
    assert res["p_value"] == pytest.approx(white_noise_test(X, lags=5)["p_value"], rel=1e-9)


def test_transposed_input_is_detected():
    assert white_noise_test(_white(seed=11).T, lags=5)["k"] == 3


def test_validates_inputs():
    X = _white(seed=12)
    with pytest.raises(ValueError, match="lags must be at least 1"):
        white_noise_test(X, lags=0)
    with pytest.raises(ValueError, match="smaller than the series length"):
        white_noise_test(X, lags=400)
    with pytest.raises(ValueError, match="must not be negative"):
        white_noise_test(X, lags=5, fitdf=-1)
    with pytest.raises(ValueError, match="must exceed fitdf"):
        white_noise_test(X, lags=3, fitdf=3)
    with pytest.raises(ValueError, match="singular"):
        white_noise_test(np.repeat(X[:, :1], 2, axis=1), lags=5)
