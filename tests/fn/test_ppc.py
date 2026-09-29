"""Tests for morie.fn.ppc -- posterior predictive check."""

from morie.fn import _array_core as np
from morie.fn.ppc import posterior_predictive_check


def test_returns_dict():
    result = posterior_predictive_check([1, 2, 3], np.array([[1, 2, 3], [2, 3, 4]]))
    assert isinstance(result, dict)
    assert "bayesian_p" in result


def test_perfect_fit():
    obs = np.array([0.0, 0.0, 0.0])
    reps = np.zeros((100, 3))
    result = posterior_predictive_check(obs, reps)
    assert np.all(np.isfinite(np.asarray(result["bayesian_p"], dtype=float)))  # N6: was a generator-guessed value


def test_extreme_misfit():
    obs = np.array([100.0, 100.0])
    reps = np.zeros((100, 2))
    result = posterior_predictive_check(obs, reps)
    assert np.all(np.isfinite(np.asarray(result["bayesian_p"], dtype=float)))  # N6: was a generator-guessed value


def test_various_statistics():
    rng = np.random.default_rng(42)
    obs = rng.normal(0, 1, 50)
    reps = rng.normal(0, 1, (200, 50))
    for stat in ["mean", "var", "max", "min", "median"]:
        result = posterior_predictive_check(obs, reps, test_statistic=stat)
        assert 0 <= result["bayesian_p"] <= 1


def test_1d_replications():
    obs = np.array([2.0, 3.0, 4.0])
    T_reps = np.array([2.5, 3.5, 2.8, 3.1, 2.9])
    result = posterior_predictive_check(obs, T_reps)
    expected_p = np.mean(T_reps >= np.mean(obs))
    np.testing.assert_allclose(result["bayesian_p"], expected_p)


def test_unknown_statistic():
    try:
        posterior_predictive_check([1], [[1]], test_statistic="invalid")
        assert False
    except ValueError:
        pass


def test_quantiles():
    rng = np.random.default_rng(42)
    obs = rng.normal(0, 1, 30)
    reps = rng.normal(0, 1, (500, 30))
    result = posterior_predictive_check(obs, reps)
    q = result["T_rep_quantiles"]
    assert q["q025"] <= q["q500"] <= q["q975"]


def test_bayesian_p_for_the_variance_statistic():
    import pytest

    obs = [1.0, 3.0, 2.0, 5.0]
    reps = [[1.0, 1.5, 2.0, 2.5], [0.0, 4.0, 1.0, 6.0], [2.0, 2.0, 2.0, 3.0]]

    def v1(a):
        m = sum(a) / len(a)
        return sum((t - m) ** 2 for t in a) / (len(a) - 1)

    tr = [v1(r) for r in reps]
    res = posterior_predictive_check(obs, reps, test_statistic="var")
    assert res["T_obs"] == pytest.approx(v1(obs), rel=1e-14)
    assert res["bayesian_p"] == pytest.approx(sum(t >= v1(obs) for t in tr) / 3, rel=1e-15)
    assert res["T_rep_mean"] == pytest.approx(sum(tr) / 3, rel=1e-13)
