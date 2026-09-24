"""Tests for shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a4e9.shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_4_equation_9."""

from morie.fn import _array_core as np

from morie.fn.shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a4e9 import (
    shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_4_equation_9,
)


def test_shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a4e9_basic():
    """Test basic functionality."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    phi_re = np.random.default_rng(42).normal(0.0, 1.0, 40)
    phi_im = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_4_equation_9(t, phi_re, phi_im, x)
    assert isinstance(result, dict)
    assert "pmf" in result


def test_shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a4e9_edge():
    """Test edge cases."""
    t = np.random.default_rng(42).normal(0.0, 1.0, 40)
    phi_re = np.random.default_rng(42).normal(0.0, 1.0, 40)
    phi_im = np.random.default_rng(42).normal(0.0, 1.0, 40)
    x = np.random.default_rng(42).normal(0.0, 1.0, 40)
    result = shailaja_r_deshmukh_akanksha_s_kashikar_probability_theory_a_chapter_4_equation_9(t, phi_re, phi_im, x)
    assert isinstance(result, dict)
