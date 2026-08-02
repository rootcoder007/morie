"""Tests for gb7381 (Gibbons shelf)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.gb7381 import gibbons_cs_null_var


def test_gb7381_basic():
    out = gibbons_cs_null_var(lambda u: u, 0.5)
    assert out["var_J"] == pytest.approx(1 / 12, abs=1e-10)


def test_gb7381_edge():
    with pytest.raises(ValueError):
        gibbons_cs_null_var(lambda u: u, 0.0)
