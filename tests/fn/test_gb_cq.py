"""Tests for gb_cq (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb_cq import gibbons_cramers_contingency


def test_gb_cq_basic():
    from morie.fn.gb1421t import gibbons_phi_cramers_v
    tbl = [[18, 7, 2], [6, 19, 5]]
    assert gibbons_cramers_contingency(tbl)["cramers_v"] == pytest.approx(
        gibbons_phi_cramers_v(tbl)["cramers_v"])


def test_gb_cq_edge():
    with pytest.raises(ValueError):
        gibbons_cramers_contingency([[0, 0], [0, 0]])
