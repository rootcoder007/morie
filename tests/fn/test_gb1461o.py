import numpy as np
"""Tests for gb1461o.gibbons_ordered_categories."""

from morie.fn import _array_core as np

from morie.fn.gb1461o import gibbons_ordered_categories


def test_gb1461o_basic():
    """Test basic functionality with the Example 14.6.2 table from the docstring."""
    # 2 x 3 contingency table from Gibbons & Chakraborti (2011), Example 14.6.2
    table = [[2, 3, 5], [4, 5, 1]]
    result = gibbons_ordered_categories(table)

    # The result is a RichResult (dict-like) — keep the existing shape check.
    assert isinstance(result, dict)
    assert "statistic" in result or "p_value" in result or "estimate" in result

    # Now assert against the documented quantities, computed independently
    # from the formula in the docstring.
    n1 = sum(table[0])  # 10
    n2 = sum(table[1])  # 10
    nn = n1 + n2        # 20

    # Column totals c_j = X_{1j} + X_{2j}.
    cs = [table[0][j] + table[1][j] for j in range(3)]
    # Default scores are the column midranks.
    acc = 0.0
    w = []
    for j in range(3):
        w.append(acc + (cs[j] + 1.0) / 2.0)
        acc += cs[j]

    t_expected = sum(w[j] * table[0][j] for j in range(3))
    wbar = sum(cs[j] * w[j] for j in range(3)) / nn
    mean_expected = n1 * wbar
    var_expected = (
        n1 * n2 / (nn * (nn - 1.0))
        * sum(cs[j] * (w[j] - wbar) ** 2 for j in range(3))
    )

    # The docstring lists these exact intermediate values for this example.
    assert result["statistic"] == t_expected
    assert result["mean"] == mean_expected
    assert result["var"] == var_expected

    # Documented numeric checks (Sec. 14.6.2 of the book).
    assert result["statistic"] == 126.0
    assert result["mean"] == 105.0

    # z = (T - mean) / sd; the book gives z ≈ 1.688 (one-sided p ≈ 0.0457).
    sd_expected = (var_expected ** 0.5) if var_expected > 0 else float("nan")
    z_expected = (t_expected - mean_expected) / sd_expected
    assert abs(result["z"] - z_expected) < 1e-12
    assert np.all(np.isfinite(np.asarray(result["z"], dtype=float)))  # N6: was a generator-guessed value

    # One-sided upper p-value should match scipy.stats.norm.sf(z).
    assert abs(result["p_value"] - (1.0 - 0.9543)) < 1e-3

    # Required keys per the docstring.
    for key in ("statistic", "mean", "var", "sd", "z",
                "p_value", "p_twosided", "scores", "n", "method"):
        assert key in result

    assert result["n"] == float(nn)
    assert list(result["scores"]) == w


def test_gb1461o_edge():
    """Test edge cases: user-supplied scores."""
    table = [[2, 3, 5], [4, 5, 1]]
    scores = [1.0, 2.0, 3.0]
    result = gibbons_ordered_categories(table, scores)

    assert isinstance(result, dict)

    # Independent recomputation with the user-provided scores.
    n1 = sum(table[0])
    n2 = sum(table[1])
    nn = n1 + n2
    cs = [table[0][j] + table[1][j] for j in range(3)]
    w = list(scores)

    t_expected = sum(w[j] * table[0][j] for j in range(3))
    wbar = sum(cs[j] * w[j] for j in range(3)) / nn
    mean_expected = n1 * wbar
    var_expected = (
        n1 * n2 / (nn * (nn - 1.0))
        * sum(cs[j] * (w[j] - wbar) ** 2 for j in range(3))
    )

    assert result["statistic"] == t_expected
    assert result["mean"] == mean_expected
    assert result["var"] == var_expected
    assert list(result["scores"]) == w
