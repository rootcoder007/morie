"""Tests for gb435 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb435 import gibbons_ks_onesided_asymp


def test_gb435_basic():
    assert gibbons_ks_onesided_asymp(1.0)["p_value"] == pytest.approx(np.exp(-2.0))


def test_gb435_edge():
    with pytest.raises(ValueError):
        gibbons_ks_onesided_asymp(-1.0)
