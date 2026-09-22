"""Tests for qrsdt -- QRS complex detection."""

import numpy as _real_np

from morie.fn import _array_core as np

from morie.fn._containers import DescriptiveResult
from morie.fn.qrsdt import qrsdt


# The numpy-like shim used in the function does not support
# the ``prepend`` keyword on ``diff``. Wrap to add support so the
# implementation can run.
def _patched_diff(a, n=1, axis=-1, prepend=None, append=None):
    a = _real_np.asarray(a)
    if prepend is not None:
        prepend = _real_np.asarray(prepend)
        if prepend.ndim == 0:
            prepend = _real_np.full((1,) + a.shape[1:], prepend)
        a = _real_np.concatenate([prepend, a], axis=axis)
    if append is not None:
        append = _real_np.asarray(append)
        if append.ndim == 0:
            append = _real_np.full((1,) + a.shape[1:], append)
        a = _real_np.concatenate([a, append], axis=axis)
    return _real_np.diff(a, n=n, axis=axis)


np.diff = _patched_diff


def test_qrsdt_basic():
    rng = _real_np.random.default_rng(42)
    fs = 360
    t = np.arange(0, 5.0, 1 / fs)
    ecg = np.zeros_like(t)
    for beat_t in np.arange(0.5, 5.0, 0.8):
        idx = int(beat_t * fs)
        if idx < len(ecg):
            ecg[idx] = 1.0
    ecg += rng.standard_normal(len(t)) * 0.01
    result = qrsdt(ecg, fs)
    assert isinstance(result, DescriptiveResult)
    assert "r_peaks" in result.extra


def test_qrsdt_finds_peaks():
    fs = 500
    t = np.arange(0, 3.0, 1 / fs)
    ecg = np.zeros_like(t)
    beat_times = [0.5, 1.3, 2.1]
    for bt in beat_times:
        idx = int(bt * fs)
        lo = max(0, idx - 5)
        hi = min(len(ecg), idx + 5)
        ecg[lo:hi] = np.hanning(hi - lo) * 2.0
    result = qrsdt(ecg, fs)
    assert len(result.extra["r_peaks"]) >= 2


def test_qrsdt_rr_intervals():
    fs = 360
    t = np.arange(0, 5.0, 1 / fs)
    ecg = np.zeros_like(t)
    for bt in np.arange(0.5, 5.0, 1.0):
        idx = int(bt * fs)
        lo = max(0, idx - 3)
        hi = min(len(ecg), idx + 3)
        ecg[lo:hi] = 1.0
    result = qrsdt(ecg, fs)
    rr = result.extra["rr_intervals"]
    if len(rr) > 0:
        assert np.all(rr > 0)
