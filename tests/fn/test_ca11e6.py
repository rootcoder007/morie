"""Tests for ca11e6.ca_chapter_11_equation_6."""

from morie.fn import _array_core as np

from morie.fn.ca11e6 import ca_chapter_11_equation_6


def test_ca11e6_basic():
    """Test basic functionality with documented 6 scalar inputs."""
    m1, m2 = 10.0, 12.0
    s1, s2 = 1.5, 1.8
    n1, n2 = 30, 25

    result = ca_chapter_11_equation_6(m1, m2, s1, s2, n1, n2)

    assert isinstance(result, dict)
    assert "t" in result

    # Independent arithmetic implementing the documented formula.
    df = (n1 - 1) + (n2 - 1)
    s_pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df)
    se = s_pooled * np.sqrt((n1 + n2) / (n1 * n2))
    expected_t = (m1 - m2) / se

    assert np.isclose(result["t"], expected_t)


def test_ca11e6_edge():
    """Test edge case: equal means yield t == 0."""
    m1, m2 = 5.5, 5.5
    s1, s2 = 2.0, 2.0
    n1, n2 = 40, 40

    result = ca_chapter_11_equation_6(m1, m2, s1, s2, n1, n2)

    assert isinstance(result, dict)
    assert "t" in result
    assert np.isclose(result["t"], 0.0)
