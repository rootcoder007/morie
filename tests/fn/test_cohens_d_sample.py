"""Tests for cohens_d_sample.cohens_d_sample."""

from morie.fn import _array_core as np

from morie.fn.cohens_d_sample import cohens_d_sample


def test_ca11e1_basic():
    """Test basic functionality."""
    m1, m2, s1, s2, n1, n2 = 10.0, 12.0, 3.0, 4.0, 50, 60
    result = cohens_d_sample(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result

    # Independent computation of the documented formula:
    # Cohen's d = (m1 - m2) / s_pooled, with
    # s_pooled = sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    expected_value = (m1 - m2) / sp
    assert np.isclose(result["value"], expected_value)

    assert result["method"] == "Weisburd et al. (2022) eq. (11.1)"


def test_ca11e1_edge():
    """Test edge cases: equal means give d = 0."""
    m1, m2, s1, s2, n1, n2 = 5.0, 5.0, 2.0, 2.0, 30, 40
    result = cohens_d_sample(m1, m2, s1, s2, n1, n2)
    assert isinstance(result, dict)
    assert "value" in result
    assert np.isclose(result["value"], 0.0)
