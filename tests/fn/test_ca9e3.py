"""Tests for ca9e3.ca_chapter_9_equation_3."""

from morie.fn import _array_core as np

from morie.fn.ca9e3 import ca_chapter_9_equation_3


def test_ca9e3_basic():
    """Test basic functionality with documented summary-statistic inputs."""
    m1, m2 = 10.0, 12.0
    s1, s2 = 3.0, 4.0
    n1, n2 = 30, 40

    result = ca_chapter_9_equation_3(m1, m2, s1, s2, n1, n2)

    # Function returns a dict subclass (RichResult) with the t-statistic under 't'
    assert isinstance(result, dict)
    assert "t" in result

    # Independent recomputation of the documented formula
    sp_num = s1**2 * (n1 - 1) + s2**2 * (n2 - 1)
    sp_den = (n1 + n2 - 2)
    sp = sp_num / sp_den
    se = np.sqrt(sp * (n1 + n2) / (n1 * n2))
    expected_t = (m1 - m2) / se

    assert np.isclose(result["t"], expected_t)


def test_ca9e3_edge():
    """Test edge case: equal group means yields t close to 0."""
    m1, m2 = 5.0, 5.0
    s1, s2 = 2.0, 2.0
    n1, n2 = 25, 25

    result = ca_chapter_9_equation_3(m1, m2, s1, s2, n1, n2)

    assert isinstance(result, dict)
    assert "t" in result
    assert np.isclose(result["t"], 0.0)
