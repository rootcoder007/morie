"""Tests for gb1431.gibbons_k2_contingency."""

import math

from morie.fn import _array_core as np

from morie.fn.gb1431 import gibbons_k2_contingency


def test_gb1431_basic():
    """Test basic functionality using the book example (Example 14.3.1).

    Book reports Q = 74.70 on 2 d.o.f. for successes 125, 81, 40
    out of group sizes 200, 200, 200.
    """
    successes = [125, 81, 40]
    ns = [200, 200, 200]
    result = gibbons_k2_contingency(successes, ns)
    assert isinstance(result, dict)
    for key in ("statistic", "df", "p_value", "phat", "props",
                "k", "n", "method"):
        assert key in result
    assert result["k"] == 3
    assert result["df"] == 2
    assert result["n"] == 600

    # Independent recomputation from the documented formula (14.3.2).
    y = [float(v) for v in successes]
    nv = [float(v) for v in ns]
    nn = sum(nv)
    ph = sum(y) / nn
    expected_q = (
        sum(y[i] ** 2 / nv[i] for i in range(len(y))) / (ph * (1.0 - ph))
        - (nn * ph / (1.0 - ph))
    )
    assert math.isclose(result["statistic"], expected_q, rel_tol=1e-12)
    assert math.isclose(result["phat"], ph, rel_tol=1e-12)
    assert result["props"] == [y[i] / nv[i] for i in range(len(y))]
    # Match the book to the reported precision.
    assert math.isclose(result["statistic"], 74.70, abs_tol=0.01)


def test_gb1431_edge():
    """Test edge case: identical proportions across two groups => Q == 0."""
    successes = [30, 30]
    ns = [100, 100]
    result = gibbons_k2_contingency(successes, ns)
    assert isinstance(result, dict)
    assert result["k"] == 2
    assert result["df"] == 1
    assert result["statistic"] == 0.0
    assert result["p_value"] == 1.0
