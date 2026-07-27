"""Tests for abind.ab_indirect_effect (Sobel)."""

import numpy as np
import pytest

from morie.fn.abind import ab_indirect_effect


def test_indirect_effect_is_the_product():
    assert ab_indirect_effect(0.5, 0.4)["estimate"] == pytest.approx(0.2)


def test_sobel_standard_error_matches_its_definition():
    a, b, sa, sb = 0.5, 0.4, 0.1, 0.08
    want = np.sqrt(b**2 * sa**2 + a**2 * sb**2)
    res = ab_indirect_effect(a, b, sa, sb)
    assert res["se"] == pytest.approx(want, rel=1e-12)
    assert res["statistic"] == pytest.approx(a * b / want, rel=1e-12)


def test_no_test_without_both_standard_errors():
    """A p-value is skipped rather than faked."""
    res = ab_indirect_effect(0.5, 0.4)
    assert res["se"] is None and res["p_value"] is None
    assert ab_indirect_effect(0.5, 0.4, se_a=0.1)["p_value"] is None


def test_a_clear_indirect_effect_is_detected():
    assert ab_indirect_effect(0.8, 0.7, 0.05, 0.05)["p_value"] < 0.001


def test_a_null_path_is_not_detected():
    assert ab_indirect_effect(0.01, 0.6, 0.2, 0.1)["p_value"] > 0.05


def test_zero_on_either_path_gives_zero_indirect_effect():
    assert ab_indirect_effect(0.0, 0.9, 0.1, 0.1)["estimate"] == pytest.approx(0.0)


def test_arrays_are_handled_elementwise():
    """So a bootstrap or posterior draw can be passed straight in."""
    a = np.array([0.2, 0.5, 0.8])
    b = np.array([0.3, 0.3, 0.3])
    res = ab_indirect_effect(a, b, np.full(3, 0.05), np.full(3, 0.05))
    assert np.allclose(res["estimate"], a * b)
    assert res["se"].shape == (3,)


def test_interval_widens_at_a_stricter_alpha():
    r95 = ab_indirect_effect(0.5, 0.4, 0.1, 0.1)
    r99 = ab_indirect_effect(0.5, 0.4, 0.1, 0.1, alpha=0.01)
    assert r99["ci_high"] - r99["ci_low"] > r95["ci_high"] - r95["ci_low"]


def test_validates_inputs():
    with pytest.raises(ValueError, match="must be finite"):
        ab_indirect_effect(np.nan, 0.4)
    with pytest.raises(ValueError, match="must not be negative"):
        ab_indirect_effect(0.5, 0.4, -0.1, 0.1)
    with pytest.raises(ValueError, match=r"alpha must lie in \(0, 1\)"):
        ab_indirect_effect(0.5, 0.4, 0.1, 0.1, alpha=1.5)
