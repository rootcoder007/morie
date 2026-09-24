"""Tests for dpitst.data_processing_inequality."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.dpitst import data_processing_inequality


def test_dpitst_basic():
    """Test basic functionality with a valid Markov chain joint pmf."""
    # Joint pmf for X->Y->Z where X is uniform on {0,1}, Y=X, and Z=Y.
    # Hence p(x,y,z) = 0.5 if x=y=z, else 0.
    pxyz = [
        [
            [0.5, 0.0],
            [0.0, 0.0],
        ],
        [
            [0.0, 0.0],
            [0.0, 0.5],
        ],
    ]
    result = data_processing_inequality(pxyz)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "ixy" in result
    assert "ixz" in result
    assert "markov_gap" in result
    assert "holds" in result
    assert "n" in result
    assert "method" in result
    assert math.isfinite(result["estimate"])
    assert math.isfinite(result["ixy"])
    assert math.isfinite(result["ixz"])
    assert math.isfinite(result["markov_gap"])
    assert isinstance(result["holds"], bool)
    assert result["holds"] is True
    assert result["n"] == 8


def test_dpitst_edge():
    """Test edge case: pxyz with zero total mass raises ValueError."""
    pxyz = [
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        [
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    ]
    with pytest.raises(ValueError):
        data_processing_inequality(pxyz)
