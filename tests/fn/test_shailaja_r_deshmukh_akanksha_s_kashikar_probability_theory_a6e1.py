"""Tests for shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a6e1.shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_6_equation_1."""

from morie.fn import _array_core as np

from morie.fn.shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a6e1 import (
    shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_6_equation_1,
)


def test_shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a6e1_basic():
    """Test basic functionality."""
    dev = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_6_equation_1(dev, k)
    assert isinstance(result, dict)
    assert "threshold" in result


def test_shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a6e1_edge():
    """Test edge cases."""
    dev = np.random.default_rng(43).normal(0.0, 1.0, (40, 3))
    k = 5
    result = shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_6_equation_1(dev, k)
    assert isinstance(result, dict)
