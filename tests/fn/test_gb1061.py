"""Tests for gb1061.gibbons_jonckheere."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gb1061 import gibbons_jonckheere


def _mann_whitney_count(x, y):
    """Compute U_{ij} = #{(a,b): x_a < y_b} with ties contributing 1/2."""
    b = 0.0
    for a in x:
        for c in y:
            if a < c:
                b += 1.0
            elif a == c:
                b += 0.5
    return b


def test_gb1061_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)
    samples = [rng.normal(0.0, 1.0, 5).tolist() for _ in range(3)]

    result = gibbons_jonckheere(samples)

    assert isinstance(result, dict)
    assert "statistic" in result
    assert "mean" in result
    assert "var" in result
    assert "z" in result
    assert "p_value" in result
    assert "k" in result
    assert "n" in result
    assert "method" in result

    # Independent recomputation of the statistic from the literature formula.
    expected_b = 0.0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            expected_b += _mann_whitney_count(samples[i], samples[j])
    assert result["statistic"] == expected_b

    # Independent recomputation of E0[B] and Var0[B] from the formulas
    # given in Sec. 10.6 of Gibbons & Chakraborti (2011).
    ns = [len(s) for s in samples]
    nn = sum(ns)
    expected_mean = (float(nn) ** 2 - sum(float(v) ** 2 for v in ns)) / 4.0
    expected_var = (
        float(nn) ** 2 * (2.0 * nn + 3.0)
        - sum(float(v) ** 2 * (2.0 * v + 3.0) for v in ns)
    ) / 72.0

    assert result["mean"] == expected_mean
    assert result["var"] == expected_var
    assert result["k"] == len(samples)
    assert result["n"] == nn
    assert 0.0 <= result["p_value"] <= 1.0


def test_gb1061_edge():
    """Test edge cases for the documented interface."""
    rng = np.random.default_rng(42)
    samples = [rng.normal(0.0, 1.0, 4).tolist() for _ in range(2)]

    result = gibbons_jonckheere(samples)
    assert isinstance(result, dict)
    assert result["k"] == 2
    assert result["n"] == sum(len(s) for s in samples)
    assert "statistic" in result
    assert "p_value" in result
