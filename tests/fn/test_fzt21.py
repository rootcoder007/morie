"""Tests for fzt21.fauzi_thm2_1_expected_kdfe."""

from morie.fn import _array_core as np

from morie.fn.fzt21 import fauzi_thm2_1_expected_kdfe


def test_fzt21_basic():
    """Test basic functionality against the documented Theorem 2.1 formula."""
    # J_h(x) and J_{ah}(x) are scalars (expected empirical CDF values at a
    # single point x). Theorem 2.1 requires a > 0 and a != 1, and jh, jah > 0
    # because the identity takes logs.
    jh = 0.6
    jah = 0.5
    a = 2.0

    result = fauzi_thm2_1_expected_kdfe(jh, jah, a)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "t1" in result and "t2" in result
    assert "a" in result
    assert result["a"] == a
    assert result["method"].startswith("geometric extrapolation")

    # Independently compute the expected quantities from the documented
    # closed-form formula: t1 = a^2/(a^2-1), t2 = -1/(a^2-1),
    # estimate = jh**t1 * jah**t2, with t1 + t2 = 1.
    expected_t1 = (a * a) / (a * a - 1.0)
    expected_t2 = -1.0 / (a * a - 1.0)
    expected_est = (jh ** expected_t1) * (jah ** expected_t2)

    assert result["t1"] == expected_t1
    assert result["t2"] == expected_t2
    assert result["estimate"] == expected_est
    assert result["t1"] + result["t2"] == 1.0


def test_fzt21_edge():
    """Test edge cases of the documented constraints (a > 0, a != 1, jh/jah > 0)."""
    # Small but positive jh/jah stay in-domain and exercise a different a.
    jh = 0.1
    jah = 0.2
    a = 0.5

    result = fauzi_thm2_1_expected_kdfe(jh, jah, a)

    assert isinstance(result, dict)
    assert "estimate" in result
    assert "t1" in result and "t2" in result
    assert result["a"] == a

    # Same closed-form check; estimate is well defined even for small values.
    expected_t1 = (a * a) / (a * a - 1.0)
    expected_t2 = -1.0 / (a * a - 1.0)
    expected_est = (jh ** expected_t1) * (jah ** expected_t2)

    assert result["t1"] == expected_t1
    assert result["t2"] == expected_t2
    assert result["estimate"] == expected_est
    assert result["t1"] + result["t2"] == 1.0
