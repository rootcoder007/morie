"""Tests for rgiir.rangayyan_iir_filter.

Spec: Rangayyan & Krishnan (2024) Sec 3.7.1 "Removal of high-frequency
noise: Butterworth lowpass filters" p.154 and Sec 3.7.2 "Removal of
low-frequency noise: Butterworth highpass filters" p.161.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsafilt import rangayyan_iir_filter

FS = 200.0


def _sig(freq, n=2048, fs=FS):
    return np.sin(2 * np.pi * freq * np.arange(n) / fs)


def test_rgiir_lowpass_keeps_the_passband_tone():
    y = np.asarray(rangayyan_iir_filter(_sig(5.0), cutoff=20.0, fs=FS)["signal"], dtype=float)
    # well inside the passband: amplitude essentially preserved
    assert np.std(y[200:-200]) == pytest.approx(np.std(_sig(5.0)), rel=0.05)


def test_rgiir_lowpass_rejects_the_stopband_tone():
    x = _sig(80.0)
    y = np.asarray(rangayyan_iir_filter(x, cutoff=10.0, order=4, fs=FS)["signal"], dtype=float)
    # 80 Hz against a 10 Hz cutoff is three octaves out; a 4th-order
    # Butterworth applied twice (filtfilt) attenuates it heavily
    assert np.std(y[200:-200]) < 0.01 * np.std(x)


def test_rgiir_highpass_removes_a_dc_offset():
    x = _sig(50.0) + 10.0
    y = np.asarray(
        rangayyan_iir_filter(x, cutoff=5.0, fs=FS, btype="high")["signal"], dtype=float
    )
    assert abs(float(np.mean(y[200:-200]))) < 0.05


def test_rgiir_is_zero_phase():
    # filtfilt is applied forward and backward, so a symmetric input stays
    # symmetric -- there is no group delay to correct for.
    n = 1024
    x = np.exp(-((np.arange(n) - n / 2) ** 2) / (2 * 20.0**2))
    y = np.asarray(rangayyan_iir_filter(x, cutoff=20.0, fs=FS)["signal"], dtype=float)
    assert int(np.argmax(y)) == pytest.approx(n // 2, abs=1)


def test_rgiir_rejects_cutoff_at_or_above_nyquist():
    # scipy raises "Digital filter critical frequencies must be 0 < Wn < 1"
    # from inside iirfilter, naming neither cutoff nor fs. With the default
    # fs=1.0 any cutoff in Hz above 0.5 trips it, which is the commonest
    # caller mistake, so the error must be stated in the caller's units.
    x = _sig(5.0)
    with pytest.raises(ValueError, match="Nyquist"):
        rangayyan_iir_filter(x, cutoff=10.0)
    with pytest.raises(ValueError, match="Nyquist"):
        rangayyan_iir_filter(x, cutoff=FS / 2, fs=FS)
    with pytest.raises(ValueError, match="Nyquist"):
        rangayyan_iir_filter(x, cutoff=0.0, fs=FS)


def test_rgiir_rejects_decreasing_band_edges():
    with pytest.raises(ValueError, match="increasing"):
        rangayyan_iir_filter(_sig(5.0), cutoff=(30.0, 10.0), fs=FS, btype="band")
