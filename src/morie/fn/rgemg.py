# morie.fn -- function file (rootcoder007/morie)
"""EMG RMS envelope -- Rangayyan & Krishnan Sec 5.6.1, eq (5.24)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_emg_rms"]


def rangayyan_emg_rms(x, window=64, fs=1.0):
    """Sliding-window RMS envelope.

    RMS[n] = sqrt( (1/W) Σ_{k=n-W+1}^{n} x[k]² ).

    Parameters
    ----------
    x : array-like
    window : int
        Window length in samples.
    fs : float
        Sampling rate (Hz, only for reporting).

    Returns
    -------
    RichResult with keys ``rms``, ``window``, ``fs``, ``mean_rms``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 5.6.1 "The RMS value", pp.283-284.
        The previous docstring cited Ch 8. Equation (5.23) is the global RMS
        over N samples; equation (5.24) is the running RMS this function
        computes,

            RMS(n) = [ (1/M) sum_{k=0}^{M-1} x^2(n-k) ]^(1/2),

        which is explicitly CAUSAL and therefore undefined for n < M-1.
    """
    x = np.asarray(x, dtype=float).ravel()
    W = int(window)
    if W < 1:
        raise ValueError("window must be >= 1")
    sq = x**2
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    rms = np.full_like(x, np.nan)
    for i in range(W - 1, x.size):
        rms[i] = np.sqrt((csum[i + 1] - csum[i + 1 - W]) / W)
    # The first W-1 samples stay NaN. Equation (5.24) is a CAUSAL window --
    # RMS(n) averages x(n-k) for k = 0..M-1 -- so it is simply undefined until
    # n = M-1; the book defines no warm-up value.
    #
    # This previously back-filled rms[:W-1] with rms[W-1], a value computed
    # from samples that lie in the FUTURE of those positions. That destroys the
    # one property eq (5.24) exists to have. Measured: a signal that is exactly
    # zero until sample 20 and active thereafter reported envelope 0.7651 at
    # sample 0, i.e. the envelope rose 20 samples BEFORE the burst. EMG onset
    # detection is the main use of an RMS envelope, so the artefact lands
    # exactly where it does the most damage.
    #
    # mean_rms already uses np.nanmean, so the warm-up was always meant to be
    # NaN; the back-fill was the anomaly.
    res = RichResult(
        title="EMG RMS envelope",
        summary_lines=[
            ("Window (samples)", W),
            ("Fs (Hz)", float(fs)),
            ("Mean RMS", float(np.nanmean(rms))),
            ("Max RMS", float(np.nanmax(rms))),
        ],
        interpretation=f"Sliding-window RMS, W={W} samples ({W / fs:.3g} s).",
        payload={"rms": rms, "window": W, "fs": float(fs), "mean_rms": float(np.nanmean(rms))},
    )
    return with_describe_pointer(res, "rgemg")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> r = rangayyan_emg_rms(rng.standard_normal(500), window=32)
# >>> r["rms"].shape == (500,)
# True


def cheatsheet():
    return "rgemg: sliding-window RMS envelope -- Rangayyan & Krishnan Sec 5.6.1"
