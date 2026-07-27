"""Tests for johbu."""

import numpy as np
import pytest

from morie.fn.johbu import joseph_bottom_up_reconciliation

_S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])


def test_johbu_basic():
    out = joseph_bottom_up_reconciliation([3.0, 4.0], _S)
    assert out["reconciled"] == pytest.approx([7.0, 3.0, 4.0])
    assert out["coherent"] is True


def test_johbu_edge():
    # an incoherent base vector is repaired, and the aggregate is used
    rec = joseph_bottom_up_reconciliation(
        None, _S, base=np.array([10.0, 3.0, 4.0]), method="ols"
    )["reconciled"]
    assert rec[0] == pytest.approx(rec[1] + rec[2])
    assert rec[0] > 7.0
    with pytest.raises(ValueError):
        joseph_bottom_up_reconciliation(None, _S, method="ols")
