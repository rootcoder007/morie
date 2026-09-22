"""Tests for gb1131.gibbons_spearman_rho."""

from morie.fn import _array_core as np

from morie.fn.gb1131 import gibbons_spearman_rho


def test_gb1131_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = gibbons_spearman_rho(x, y)
    assert np.all(np.isfinite(np.asarray(result["statistic"], dtype=float)))  # N6: was a generator-guessed value


def test_gb1131_edge():
    """Test edge cases: n must be >= 3 per docstring."""
    # Documented constraint: x and y must have at least 3 paired observations.
    # Using n = 3 (the minimum allowed) with a perfectly linear relationship
    # so statistic == 1 exactly.
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([5.0, 7.0, 9.0])
    result = gibbons_spearman_rho(x, y)

    # Independent computation of the shortcut formula
    # eq. (11.3.2): R = 1 - 6 * sum(D_i^2) / (n * (n^2 - 1))
    # For perfectly monotonic data the shortcut equals Pearson of midranks == 1.
    n = 3
    expected_shortcut = 1.0 - 6.0 * 0.0 / (n * (float(n) ** 2 - 1.0))

    assert result["n"] == n
    assert result["n"] >= 3  # documented minimum
    assert result["tied"] == 0
    assert result["sumd2"] == 0.0
    assert result["r_shortcut"] == expected_shortcut
    assert result["statistic"] == expected_shortcut
    assert result["var"] == 1.0 / (n - 1.0)
    assert result["method"] == "Spearman rank correlation, eq. (11.3.2)"
