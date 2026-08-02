"""Tests for cipsc.caliper_psm."""

from morie.fn import _array_core as np
import pytest

from morie.fn.cipsc import caliper_psm


def test_cipsc_basic():
    e = np.array([0.6, 0.59, 0.3, 0.05])
    T = np.array([1, 0, 1, 0])
    result = caliper_psm(e, T, caliper=0.3)
    # logit gaps: (0.6, 0.59) = 0.04 < 0.3 matched; (0.3, 0.05) = 2.1 unmatched
    assert {tuple(p) for p in result["matched_idx"]} == {(0, 1)}
    assert result["n_matched"] == 1
    assert result["n_treated"] == 2


def test_cipsc_edge():
    # default caliper = 0.2 * sd(logit(e))
    e = np.array([0.2, 0.4, 0.6, 0.8])
    T = np.array([1, 0, 1, 0])
    lg = np.log(e / (1 - e))
    result = caliper_psm(e, T)
    assert result["caliper"] == pytest.approx(0.2 * lg.std(ddof=1))
    with pytest.raises(ValueError):
        caliper_psm(e, T, caliper=-1.0)
