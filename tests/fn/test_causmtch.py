"""Tests for causmtch.causal_pair_matching."""

import numpy as np
import pytest

from morie.fn.causmtch import causal_pair_matching


def test_causmtch_basic():
    ps = np.array([0.7, 0.69, 0.4, 0.41, 0.2])
    treat = np.array([1, 0, 1, 0, 0])
    y = np.array([3.0, 1.0, 2.0, 1.5, 0.0])
    result = causal_pair_matching(ps, treat, y=y)
    pairs = {tuple(p) for p in result["matched_idx"]}
    assert pairs == {(0, 1), (2, 3)}  # nearest neighbours on logit scale
    assert result["att"] == pytest.approx(((3.0 - 1.0) + (2.0 - 1.5)) / 2)


def test_causmtch_edge():
    with pytest.raises(ValueError):
        causal_pair_matching([0.5, 0.5], [1, 1])  # no controls
    with pytest.raises(ValueError):
        causal_pair_matching([1.2, 0.5], [1, 0])  # ps outside (0, 1)
