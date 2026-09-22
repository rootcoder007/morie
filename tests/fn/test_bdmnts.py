"""Tests for bdmnts.bound_monot_inst."""

from morie.fn import _array_core as np

from morie.fn.bdmnts import bound_monot_inst


def test_bdmnts_basic():
    """Test basic functionality."""
    lower = [0.0, 0.1, 0.2, 0.3, 0.4]
    upper = [0.5, 0.6, 0.7, 0.8, 0.9]
    prob = [0.2, 0.2, 0.2, 0.2, 0.2]
    result = bound_monot_inst(lower, upper, prob)

    # Recompute expected monotone-instrument bounds directly from the
    # formula in the docstring using only plain arithmetic on the inputs.
    p = [v / sum(prob) for v in prob]
    Lv = []
    run = lower[0]
    for i in range(len(lower)):
        run = max(run, lower[i])
        Lv.append(run)
    Uv = [0.0] * len(upper)
    run = upper[-1]
    for i in range(len(upper) - 1, -1, -1):
        run = min(run, upper[i])
        Uv[i] = run
    expected_lb = sum(p[i] * Lv[i] for i in range(len(prob)))
    expected_ub = sum(p[i] * Uv[i] for i in range(len(prob)))

    assert isinstance(result, dict)
    assert "lower" in result
    assert "upper" in result
    assert "width" in result
    assert "lowerv" in result
    assert "upperv" in result
    assert "prob" in result
    assert "k" in result

    assert np.isclose(result["lower"], expected_lb)
    assert np.isclose(result["upper"], expected_ub)
    assert np.isclose(result["width"], expected_ub - expected_lb)
    assert np.allclose(result["lowerv"], Lv)
    assert np.allclose(result["upperv"], Uv)
    assert np.allclose(result["prob"], p)
    assert result["k"] == len(prob)


def test_bdmnts_edge():
    """Test edge cases."""
    # Single instrument value: per-value bounds are themselves the bounds.
    lower = [0.2]
    upper = [0.7]
    prob = [1.0]
    result = bound_monot_inst(lower, upper, prob)

    assert isinstance(result, dict)
    assert result["k"] == 1
    assert np.isclose(result["lower"], 0.2)
    assert np.isclose(result["upper"], 0.7)
    assert np.isclose(result["width"], 0.5)
    assert np.allclose(result["lowerv"], [0.2])
    assert np.allclose(result["upperv"], [0.7])
    assert np.allclose(result["prob"], [1.0])
