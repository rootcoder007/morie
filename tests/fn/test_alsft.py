"""Tests for alsft.alammar_setfit_twostep."""

from morie.fn.alsft import alammar_setfit_twostep


def test_alsft_basic():
    out = alammar_setfit_twostep([[0, 0], [0.1, 0], [5, 5], [5.1, 5]],
                                 [0, 0, 1, 1])
    assert out["n_positive"] == 2


def test_alsft_edge():
    import pytest
    with pytest.raises(ValueError, match="two classes"):
        alammar_setfit_twostep([[0, 0], [1, 1]], [0, 0])
