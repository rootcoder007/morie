"""Tests for ksr042 (Kosorok shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr042 import kosorok_ch2_functional_delta_method


def test_ksr042_basic():
    out = kosorok_ch2_functional_delta_method(lambda x: x**2, np.array(2.01),
                                              np.array(2.0), r_n=100.0)
    assert float(out["derivative"]) == pytest.approx(0.04, abs=1e-6)


def test_ksr042_edge():
    with pytest.raises(ValueError):
        kosorok_ch2_functional_delta_method(lambda x: x**2, 2.0, 2.0, r_n=0.0)
