"""Tests for causdid2.causal_did_2x2."""

from morie.fn import _array_core as np
import pytest

from morie.fn.causdid2 import causal_did_2x2


def test_causdid2_basic():
    # cells: T0=1, T1=4, C0=2, C1=3 -> ATT = 3 - 1 = 2
    y = np.array([1.0, 1.0, 4.0, 4.0, 2.0, 2.0, 3.0, 3.0])
    T = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    P = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    result = causal_did_2x2(y, T, P)
    assert result["att"] == pytest.approx(2.0)
    assert result["cell_means"]["T1"] == pytest.approx(4.0)


def test_causdid2_edge():
    with pytest.raises(ValueError):
        causal_did_2x2([1.0, 2.0], [0.5, 1.0], [0, 1])  # non-binary group
    with pytest.raises(ValueError):
        causal_did_2x2([1.0, 2.0], [1, 1], [0, 1])  # empty control cells
