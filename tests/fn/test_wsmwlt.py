"""Tests for wsmwlt.wasserman_wavelet_smooth (Haar, universal threshold)."""

import math

import pytest

from morie.fn.wsmwlt import wasserman_wavelet_smooth


def _haar(x):
    a, ds = list(x), []
    while len(a) > 1:
        ds.append([(a[2 * k] - a[2 * k + 1]) / math.sqrt(2) for k in range(len(a) // 2)])
        a = [(a[2 * k] + a[2 * k + 1]) / math.sqrt(2) for k in range(len(a) // 2)]
    return a, ds


def _ihaar(a, ds):
    for d in reversed(ds):
        out = []
        for s, t in zip(a, d):
            out += [(s + t) / math.sqrt(2), (s - t) / math.sqrt(2)]
        a = out
    return a


def test_wsmwlt_basic():
    """lambda = sigma sqrt(2 log n); details hard-thresholded, the
    approximation kept, the signal rebuilt by the inverse Haar transform."""
    y = [math.sin(0.4 * k) * 3 + 0.2 * math.cos(5.1 * k) for k in range(32)]
    lam = 0.5 * math.sqrt(2 * math.log(32))
    a, ds = _haar(y)
    kept = [[v if abs(v) > lam else 0.0 for v in d] for d in ds]
    r = wasserman_wavelet_smooth(y, sigma=0.5)
    assert r["threshold"] == pytest.approx(lam, rel=1e-15)
    assert r["estimate"] == pytest.approx(_ihaar(a, kept), abs=1e-12)
    assert r["n_kept"] == sum(1 for d in kept for v in d if v != 0.0)
    assert r["n_detail"] == 31


def test_wsmwlt_edge():
    """sigma defaults to the centred MAD of the finest details / 0.67449; non-Haar wavelets
    and non-power-of-two lengths raise."""
    y = [math.sin(0.4 * k) * 3 + 0.2 * math.cos(5.1 * k) for k in range(32)]
    _, ds = _haar(y)
    # the centred MAD, median |d - median d| (R's mad(), as wavethresh
    # uses it), over the 16 finest details
    med = 0.5 * sum(sorted(ds[0])[7:9])
    fin = sorted(abs(v - med) for v in ds[0])
    mad = 0.5 * (fin[7] + fin[8])
    assert wasserman_wavelet_smooth(y)["sigma_used"] == pytest.approx(mad / 0.6744897501960817, rel=1e-12)
    with pytest.raises(ValueError):
        wasserman_wavelet_smooth(y, wavelet="db4")
    with pytest.raises(ValueError):
        wasserman_wavelet_smooth(y[:30], sigma=1.0)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.wsmwlt as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
