"""Tests for rgfir.rangayyan_fir_filter.

Spec sources
------------
The windowed-sinc design this function implements is NOT in Rangayyan's
Ch. 3 -- that chapter's frequency-domain filtering is Butterworth (IIR),
Sec. 3.7.1-3.7.3 of the 2024 edition, verified against the book's own table
of contents. The authoritative specification for what this function actually
does is the SciPy reference documentation for ``scipy.signal.firwin``:

  - "raises ValueError if any value in cutoff is ... greater than or equal
    to fs/2"
  - with ``scale=True`` (default) it normalises "the coefficients so that
    the frequency response is exactly unity" at DC for a lowpass

Both are pinned below. No worked example is transcribed here because no book
in the library carries one for this design; the identity tests carry the
weight instead.
"""

import numpy as np
import pytest

from morie.fn.rgfir import rangayyan_fir_filter


def test_returns_documented_keys():
    """The result carries its documented contract.

    The generated test asserted a generic ``"estimate"`` key that this
    function never promised -- it returns signal/taps/order/cutoff/fs. The
    test was wrong, not the function, so this pins the real contract.
    """
    x = np.random.default_rng(42).normal(0, 1, 200)
    result = rangayyan_fir_filter(x, cutoff=0.1, order=21, fs=1.0)
    for key in ("signal", "taps", "order", "cutoff", "fs"):
        assert key in result, f"missing documented key {key!r}"
    assert np.asarray(result["signal"]).shape == x.shape


def test_rejects_cutoff_at_or_above_nyquist():
    """cutoff >= fs/2 must raise, matching scipy.signal.firwin.

    The old implementation clipped fc into (0, 1), so cutoff=10 Hz at
    fs=1 Hz -- twenty times Nyquist -- silently returned a near-Nyquist
    filter. That is a caller error being converted into a plausible wrong
    answer, which is exactly what this audit exists to remove.
    """
    x = np.random.default_rng(0).normal(0, 1, 100)
    with pytest.raises(ValueError, match="Nyquist"):
        rangayyan_fir_filter(x, cutoff=10.0, order=21, fs=1.0)
    with pytest.raises(ValueError, match="Nyquist"):
        rangayyan_fir_filter(x, cutoff=0.5, order=21, fs=1.0)  # exactly Nyquist
    with pytest.raises(ValueError, match="Nyquist"):
        rangayyan_fir_filter(x, cutoff=0.0, order=21, fs=1.0)


def test_identity_dc_gain_is_unity():
    """sum(taps) == 1: unity gain at DC.

    scipy.signal.firwin with scale=True normalises "the coefficients so that
    the frequency response is exactly unity" at DC for a lowpass. The DC
    response of an FIR filter is H(0) = sum_n h[n], so the coefficients must
    sum to 1. A scaling or normalisation slip breaks this while still
    producing a filter-shaped array of numbers.
    """
    x = np.random.default_rng(1).normal(0, 1, 300)
    for cutoff, fs in ((0.1, 1.0), (25.0, 250.0), (2.0, 100.0)):
        taps = np.asarray(
            rangayyan_fir_filter(x, cutoff=cutoff, order=31, fs=fs)["taps"]
        )
        assert np.isclose(taps.sum(), 1.0, atol=1e-12), (
            f"DC gain {taps.sum():.6f} != 1 for cutoff={cutoff}, fs={fs}"
        )


def test_identity_linear_phase_symmetric_taps():
    """A Type I linear-phase FIR has symmetric coefficients, h[n] = h[N-1-n].

    The implementation forces the tap count odd for exactly this reason, so
    the symmetry is the property that confirms it succeeded.
    """
    x = np.random.default_rng(2).normal(0, 1, 200)
    taps = np.asarray(rangayyan_fir_filter(x, cutoff=0.2, order=41, fs=1.0)["taps"])
    assert taps.size % 2 == 1
    assert np.allclose(taps, taps[::-1], atol=1e-12)


def test_identity_lowpass_attenuates_above_cutoff():
    """A lowpass keeps a below-cutoff tone and suppresses one above it.

    This is the behavioural claim the function's name makes; without it every
    other assertion here is satisfied by a filter that does nothing useful.
    """
    fs, n = 500.0, 2000
    t = np.arange(n) / fs
    low = np.sin(2 * np.pi * 5.0 * t)
    high = np.sin(2 * np.pi * 200.0 * t)
    out_low = np.asarray(
        rangayyan_fir_filter(low, cutoff=50.0, order=101, fs=fs)["signal"]
    )
    out_high = np.asarray(
        rangayyan_fir_filter(high, cutoff=50.0, order=101, fs=fs)["signal"]
    )
    keep = slice(200, -200)  # ignore edge transients
    assert out_low[keep].std() > 0.9 * low[keep].std()
    assert out_high[keep].std() < 0.05 * high[keep].std()


def test_even_order_is_promoted_to_odd():
    """Documented behaviour: even tap counts become odd (Type I)."""
    x = np.random.default_rng(3).normal(0, 1, 200)
    assert rangayyan_fir_filter(x, cutoff=0.1, order=20, fs=1.0)["order"] == 21
