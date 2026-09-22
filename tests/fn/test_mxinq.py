"""Tests for morie.fn.mxinq — maximal inequality bound."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mxinq import mxinq


def test_basic_output():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = mxinq(x, n_boot=50, seed=7)
    assert result["expected_sup"] > 0
    assert result["dkw_bound"] > 0
    assert result["n"] == 200


def test_more_data_smaller_bound():
    rng = np.random.default_rng(42)
    r1 = mxinq(rng.standard_normal(50), n_boot=20, seed=1)
    r2 = mxinq(rng.standard_normal(500), n_boot=20, seed=1)
    assert r2["dkw_bound"] < r1["dkw_bound"]


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mxinq(np.array([]))
