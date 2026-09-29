"""Tests for morie.fn.bfmi -- BFMI diagnostic."""

from morie.fn import _array_core as np
from morie.fn.bfmi import bayesian_fmi


def test_returns_dict():
    result = bayesian_fmi([1.0, 2.0, 1.5, 2.5, 1.8])
    assert isinstance(result, dict)
    assert "bfmi" in result


def test_good_mixing():
    rng = np.random.default_rng(42)
    energy = rng.normal(0, 1, 1000)
    result = bayesian_fmi(energy)
    assert result["bfmi"] > 0.3
    assert result["adequate"] is True


def test_poor_mixing():
    energy = np.cumsum(np.ones(100))
    result = bayesian_fmi(energy)
    assert result["adequate"] is False


def test_constant_energy():
    result = bayesian_fmi(np.ones(100))
    assert np.all(np.isfinite(np.asarray(result["bfmi"], dtype=float)))  # N6: was a generator-guessed value


def test_too_short():
    try:
        bayesian_fmi([1.0, 2.0])
        assert False
    except ValueError:
        pass


def test_bfmi_is_betancourts_ratio_of_sums():
    import pytest

    e = [3.2, 1.1, 4.7, 2.2, 2.9, 5.1, 0.4, 3.3]
    n = len(e)
    m = sum(e) / n
    num = sum((e[i] - e[i - 1]) ** 2 for i in range(1, n))
    den = sum((v - m) ** 2 for v in e)
    r = bayesian_fmi(e)
    assert r["bfmi"] == pytest.approx(num / den, rel=1e-13)
    assert r["energy_var"] == pytest.approx(den / (n - 1), rel=1e-13)
    assert r["transition_var"] == pytest.approx(num / (n - 1), rel=1e-13)
