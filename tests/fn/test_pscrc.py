"""Tests for pscrc."""

from morie.fn import _array_core as np
import pytest

from morie.fn.pscrc import pscl_rollcall


def test_pscrc_basic():
    raw = np.array([[1, 1], [1, 1], [1, 0], [1, 1]])
    out = pscl_rollcall(raw, lop=0.1)
    assert bool(out["keep"][0]) is False  # unanimous
    assert bool(out["keep"][1]) is True


def test_pscrc_edge():
    raw = np.array([[1, 9], [0, 0]])
    out = pscl_rollcall(raw, missing=(9,))
    assert np.isnan(out["votes"][0, 1])
    with pytest.raises(ValueError):
        pscl_rollcall(raw, lop=0.6)
