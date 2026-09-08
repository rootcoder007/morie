"""Tests for altsd.alammar_tsdae_objective."""

from morie.fn.altsd import alammar_tsdae_objective


def test_altsd_basic():
    out = alammar_tsdae_objective(list("abcdef"), seed=2)
    assert len(out["corrupted"]) + len(out["deleted"]) == 6


def test_altsd_edge():
    import pytest
    with pytest.raises(ValueError, match="delete_ratio"):
        alammar_tsdae_objective(["a"], delete_ratio=1.5)
