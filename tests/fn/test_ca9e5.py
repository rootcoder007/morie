"""Tests for ca9e5.ca_chapter_9_equation_5."""

from morie.fn import _array_core as np

from morie.fn.ca9e5 import ca_chapter_9_equation_5


def _compute_ms_between(groups):
    """Compute MS_between from the documented formula using plain arithmetic.

    MS_between = sum_j n_j * (ybar_j - ybar)^2 / (a - 1)
    """
    a = len(groups)
    sizes = [len(g) for g in groups]
    means = [float(np.mean(g)) for g in groups]
    n_total = sum(sizes)
    grand_mean = sum(s * m for s, m in zip(sizes, means)) / n_total
    num = sum(s * (m - grand_mean) ** 2 for s, m in zip(sizes, means))
    return num / (a - 1)


def test_ca9e5_basic():
    """Test basic functionality with a documented multi-group input."""
    rng = np.random.default_rng(42)
    groups = [rng.normal(0, 1, 30), rng.normal(1, 1, 30), rng.normal(2, 1, 30)]
    result = ca_chapter_9_equation_5(groups)

    assert isinstance(result, dict)
    # Headline value lives under 'ms_between' per the docstring / formula.
    assert "ms_between" in result
    # Independent recomputation from the documented formula.
    expected = _compute_ms_between(groups)
    assert abs(result["ms_between"] - expected) < 1e-9


def test_ca9e5_edge():
    """Test edge cases with the minimum valid input (2 groups, >= 2 obs each)."""
    rng = np.random.default_rng(42)
    groups = [rng.normal(0, 1, 10), rng.normal(5, 1, 10)]
    result = ca_chapter_9_equation_5(groups)

    assert isinstance(result, dict)
    assert "ms_between" in result
    expected = _compute_ms_between(groups)
    assert abs(result["ms_between"] - expected) < 1e-9
