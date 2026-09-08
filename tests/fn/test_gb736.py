"""Tests for gb736 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb736 import gibbons_linrank_sym_special


def test_gb736_basic():
    out = gibbons_linrank_sym_special(8)
    assert out["palindromic"] is True and out["symmetric"] is True


def test_gb736_edge():
    with pytest.raises(ValueError):
        gibbons_linrank_sym_special(7)  # odd N genuinely skewed
