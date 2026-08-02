"""Tests for ksr065 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr065 import kosorok_ch3_efficient_influence_general


def test_ksr065_basic():
    out = kosorok_ch3_efficient_influence_general(np.array([[2.0, 0.0], [0.0, 4.0]]),
                                                 np.array([2.0, 8.0]))
    assert out["chi"] == pytest.approx([1.0, 2.0])
    assert out["consistent"] is True


def test_ksr065_edge():
    # an inconsistent system means the parameter is NOT pathwise
    # differentiable -- reported, not silently least-squared away
    bad = kosorok_ch3_efficient_influence_general(np.ones((2, 2)), np.array([1.0, 5.0]))
    assert bad["consistent"] is False
