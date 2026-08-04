"""Tests for rng196.rangayyan_ch4_dicrotic_notch_second_derivative."""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsaqrs import rangayyan_ch4_dicrotic_notch_second_derivative


def test_rng196_basic():
    # Eq. (4.22), p. 228: taps (2, -1, -2, -1, 2) sum to zero
    spike = np.zeros(9)
    spike[4] = 1.0
    result = rangayyan_ch4_dicrotic_notch_second_derivative(spike)
    assert result["p"][2:7] == pytest.approx([2.0, -1.0, -2.0, -1.0, 2.0])
    flat = rangayyan_ch4_dicrotic_notch_second_derivative(np.ones(9))
    assert flat["p"][2:7] == pytest.approx(np.zeros(5), abs=1e-12)


def test_rng196_edge():
    with pytest.raises(ValueError):
        rangayyan_ch4_dicrotic_notch_second_derivative([1.0, 2.0, 3.0])  # < 5 samples
    spike = np.zeros(9)
    spike[4] = 1.0
    causal = rangayyan_ch4_dicrotic_notch_second_derivative(spike, causal=True)["p"]
    noncausal = rangayyan_ch4_dicrotic_notch_second_derivative(spike)["p"]
    assert causal[4:9] == pytest.approx(noncausal[2:7])  # two-sample delay
