"""Tests for grjll.geron_johnson_lindenstrauss_bound."""

import math

import pytest

from morie.fn import _array_core as np
from morie.fn.grjll import geron_johnson_lindenstrauss_bound


def test_grjll_basic():
    """Test basic functionality with scalar eps."""
    result = geron_johnson_lindenstrauss_bound(10000, 0.1)
    assert isinstance(result, dict)
    for key in ("min_dimension", "estimate", "n_samples", "eps",
                "denominator", "method", "n"):
        assert key in result
    assert result["n_samples"] == 10000
    assert result["n"] == 10000

    # min_dimension may surface as an int or as a single-element list
    md = result["min_dimension"]
    if isinstance(md, list):
        assert len(md) == 1
        md_val = md[0]
    else:
        md_val = md
    assert isinstance(md_val, int)
    assert md_val > 0

    # estimate mirrors min_dimension
    est = result["estimate"]
    if isinstance(est, list):
        assert len(est) == 1
        est_val = est[0]
    else:
        est_val = est
    assert isinstance(est_val, int)
    assert est_val > 0

    # denominator must be finite
    denom = result["denominator"]
    if isinstance(denom, list):
        assert len(denom) == 1
        denom_val = denom[0]
    else:
        denom_val = denom
    assert math.isfinite(denom_val)


def test_grjll_edge():
    """Test edge case with eps as an array of values."""
    result = geron_johnson_lindenstrauss_bound(100, [0.1, 0.2, 0.5])
    assert isinstance(result, dict)
    assert isinstance(result["min_dimension"], list)
    assert len(result["min_dimension"]) == 3
    for d in result["min_dimension"]:
        assert isinstance(d, int)
        assert d > 0
