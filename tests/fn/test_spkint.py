"""Tests for spkint.spike_information (direct method, Strong et al. 1998)."""

import math

import pytest

from morie.fn.spkint import spike_information


def _H(counts):
    m = sum(counts)
    return -sum(c / m * math.log2(c / m) for c in counts if c)


def test_spkint_basic():
    """Equal-count bins at the sorted-sample quantiles; I = H(R) - sum_s
    P(s) H(R | s).  Hand-worked: 8 trials, 2 bins split at the 4th order
    statistic, stimulus 1 responses all low except one."""
    spike = [0, 1, 1, 2, 3, 4, 5, 6]
    stim = [1, 1, 1, 2, 1, 2, 2, 2]
    # edge = sorted[4-1] = 2 -> low bin {0,1,1,2}, high {3,4,5,6}
    # stim 1: responses 0,1,1,3 -> low 3, high 1; stim 2: 2,4,5,6 -> 1, 3
    ht = _H([4, 4])
    hn = 0.5 * _H([3, 1]) + 0.5 * _H([1, 3])
    r = spike_information(spike, stim, nbins=2)
    assert r["h_total"] == pytest.approx(ht, abs=1e-15)
    assert r["h_noise"] == pytest.approx(hn, abs=1e-15)
    assert r["information"] == pytest.approx(ht - hn, abs=1e-15)
    assert r["n_per_cell"] == 2.0


def test_spkint_edge():
    """Unbalanced classes are weighted by frequency, not equally;
    a perfectly informative response carries H(S) bits; bad inputs raise."""
    spike = [0, 0, 0, 0, 0, 0, 1, 1]
    stim = [1, 1, 1, 1, 1, 1, 2, 2]
    r = spike_information(spike, stim, nbins=2)
    assert r["information"] == pytest.approx(_H([6, 2]), abs=1e-15)
    with pytest.raises(ValueError):
        spike_information([0, 1, 2, 3], [1.5, 1, 2, 2])
    with pytest.raises(ValueError):
        spike_information([0, 1, 2, 3], [1, 1, 1, 1])
    with pytest.raises(ValueError):
        spike_information([0, 1, 2], [1, 2, 1])
    with pytest.raises(ValueError):
        spike_information([0, 1, 2, 3], [1, 2, 1, 2], nbins=5)
