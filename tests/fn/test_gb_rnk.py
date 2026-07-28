"""Tests for gb_rnk (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_rnk import gibbons_rank_def


def test_gb_rnk_basic():
    out = gibbons_rank_def([3.0, 1.0, 4.0])
    assert list(out["ranks"]) == [2, 1, 3]


def test_gb_rnk_edge():
    with pytest.raises(ValueError):
        gibbons_rank_def([1.0, 1.0])
