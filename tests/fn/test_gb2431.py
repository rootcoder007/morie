"""Tests for gb2431 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb2431 import gibbons_binomial_beta_link


def test_gb2431_basic():
    assert gibbons_binomial_beta_link(0.4, 3, 9)["agree"] is True


def test_gb2431_edge():
    with pytest.raises(ValueError):
        gibbons_binomial_beta_link(0.5, 0, 5)
