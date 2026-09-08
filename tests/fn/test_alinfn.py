"""Tests for alinfn.alammar_infonce_loss."""

from morie.fn.alinfn import alammar_infonce_loss


def test_alinfn_basic():
    out = alammar_infonce_loss([1.0, 0.0], [1.0, 0.0], [[0.0, 1.0]],
                               tau=1.0)
    assert out["estimate"] > 0


def test_alinfn_edge():
    import pytest
    with pytest.raises(ValueError, match="temperature"):
        alammar_infonce_loss([1.0], [1.0], [[1.0]], tau=0.0)
