# morie.fn -- function file (rootcoder007/morie)
"""LMS adaptive noise canceller -- Rangayyan & Krishnan Sec 3.10.2."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_adaptive_filter"]


def rangayyan_adaptive_filter(x, reference, mu=0.01, order=16):
    """LMS adaptive noise canceller (Widrow-Hoff).

    For each sample::

        y[n] = w[n].T r_vec[n]
        e[n] = x[n] - y[n]
        w[n+1] = w[n] + 2 μ e[n] r_vec[n]

    Cleaned signal is ``e``; estimated noise is ``y``.

    Parameters
    ----------
    x : array-like
        Primary signal (target + correlated noise).
    reference : array-like
        Reference noise (same length).
    mu : float
        Step size.
    order : int
        Number of taps.

    Returns
    -------
    RichResult with keys ``signal`` (=e), ``noise_estimate`` (=y),
    ``weights``, ``mu``, ``order``.

    References
    ----------
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 3.10.2 "The least-mean-squares
        adaptive filter", p.184. The previous docstring cited Ch 11.
    Widrow, B., & Stearns, S. D. (1985). *Adaptive Signal Processing*.
        Prentice-Hall.
    """
    x = np.asarray(x, dtype=float).ravel()
    r = np.asarray(reference, dtype=float).ravel()
    if x.size != r.size:
        raise ValueError("x and reference must have equal length.")
    M = int(order)
    N = x.size
    w = np.zeros(M)
    y = np.zeros(N)
    e = np.zeros(N)
    # Start at n = 0 with a zero-padded reference history rather than at
    # n = M-1. The old loop left y[0..M-2] and e[0..M-2] at zero, so the first
    # M-1 samples of the returned signal were not the cancelled input -- they
    # were nothing at all. At the default order=16 that silently destroyed 15
    # samples, and it broke the identity that defines a noise canceller:
    # signal + noise_estimate must reconstruct the primary input everywhere.
    # Zero-padding the history is the standard LMS start-up and keeps the
    # filter adapting from the first sample.
    for n in range(N):
        seg = r[max(0, n - M + 1) : n + 1][::-1]
        rv = np.zeros(M)
        rv[: seg.size] = seg
        y[n] = float(w @ rv)
        e[n] = x[n] - y[n]
        w = w + 2.0 * mu * e[n] * rv
    res = RichResult(
        title="LMS adaptive noise canceller",
        summary_lines=[
            ("Taps", M),
            ("μ", float(mu)),
            ("Residual MSE", float(np.mean(e**2))),
        ],
        interpretation=f"LMS order {M}, μ={mu}; residual MSE {float(np.mean(e**2)):.4g}.",
        payload={"signal": e, "noise_estimate": y, "weights": w, "mu": float(mu), "order": M},
    )
    return with_describe_pointer(res, "rgadp")


# CANONICAL TEST
# >>> rng = np.random.default_rng(0)
# >>> n = rng.standard_normal(500)
# >>> x = np.sin(2*np.pi*np.arange(500)/50.0) + n
# >>> r = rangayyan_adaptive_filter(x, reference=n, mu=0.01, order=8)
# >>> r["signal"].shape == (500,)
# True


def cheatsheet():
    return "rgadp: LMS adaptive noise canceller -- Rangayyan & Krishnan Sec 3.10.2"
