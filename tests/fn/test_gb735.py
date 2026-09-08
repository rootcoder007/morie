"""Tests for gb735 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb735 import gibbons_linrank_sym_equal


def test_gb735_basic():
    ugly = np.array([0.0, 1.0, 1.5, 7.0, 7.2, 11.0])
    assert gibbons_linrank_sym_equal(ugly, 3, 3)["symmetric"] is True


def test_gb735_edge():
    ugly = np.array([0.0, 1.0, 1.5, 7.0, 7.2, 11.0])
    assert gibbons_linrank_sym_equal(ugly, 2, 4)["symmetric"] is False
    with pytest.raises(ValueError):
        gibbons_linrank_sym_equal(ugly, 2, 3)  # length mismatch
