"""Tests for gb1241t (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb1241t import gibbons_concordance_w_ties


def test_gb1241t_basic():
    R = np.tile(np.arange(1, 6), (3, 1))
    assert gibbons_concordance_w_ties(R)["W"] == pytest.approx(1.0)  # tie-free == plain


def test_gb1241t_edge():
    tied = np.array([[1.5, 1.5, 3.0], [1.0, 2.0, 3.0]])
    assert gibbons_concordance_w_ties(tied)["tie_sum"] > 0
    with pytest.raises(ValueError):
        gibbons_concordance_w_ties(np.ones((2, 3)))  # every judge all-tied
