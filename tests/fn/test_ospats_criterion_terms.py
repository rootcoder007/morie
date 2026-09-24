"""Tests for ospats_criterion_terms.ospats_criterion_terms."""

import math

from morie.fn import _array_core as np
from morie.fn.ospats_criterion_terms import ospats_criterion_terms


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e12_basic():
    w_h = np.linspace(0.1, 1.0, 10)
    s_h = np.linspace(0.5, 2.0, 10)
    expected = (sum(w * s for w, s in zip(w_h, s_h))) ** 2
    result = ospats_criterion_terms(w_h, s_h)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isclose(result["value"], expected)


def test_the_r_series_dick_j_brus_spatial_sampling_with_r13e12_edge():
    w_h = [1.0]
    s_h = [0.5]
    expected = (sum(w * s for w, s in zip(w_h, s_h))) ** 2
    result = ospats_criterion_terms(w_h, s_h)
    assert isinstance(result, dict)
    assert "value" in result
    assert math.isclose(result["value"], expected)
    assert math.isfinite(result["value"])
    assert result["value"] >= 0
