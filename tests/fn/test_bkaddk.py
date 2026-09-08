"""Tests for bkaddk.burkov_add_k_smoothing."""

from morie.fn.bkaddk import burkov_add_k_smoothing


def test_bkaddk_basic():
    assert burkov_add_k_smoothing(0, 0, 4, k=0.5)["estimate"] == 0.25


def test_bkaddk_edge():
    import pytest
    with pytest.raises(ValueError, match="k must be positive"):
        burkov_add_k_smoothing(1, 2, 3, k=0.0)
