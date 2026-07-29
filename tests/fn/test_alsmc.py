"""Tests for alsmc.alammar_simcse_dropout_aug."""

from morie.fn.alsmc import alammar_simcse_dropout_aug


def test_alsmc_basic():
    A = [[1.0, 0.0], [0.0, 1.0]]
    out = alammar_simcse_dropout_aug(A, A, tau=1.0)
    assert out["estimate"] > 0


def test_alsmc_edge():
    import pytest
    with pytest.raises(ValueError, match="align"):
        alammar_simcse_dropout_aug([[1.0]], [[1.0], [2.0]])
