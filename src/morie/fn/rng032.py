# morie.fn -- function file (rootcoder007/morie)
"""Causal continuous-time convolution form (lower limit 0, upper limit t)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_causal_convolution"]


def rangayyan_ch3_causal_convolution(x, h, dt=1.0):
    r"""Causal continuous convolution
    :math:`y(t) = \int_0^t x(\tau) h(t-\tau)\, d\tau`.

    Evaluated on the sample grid by the trapezoidal rule for every
    upper limit t, so ``y[i]`` approximates the integral up to
    ``t = i * dt``. Both signals are treated as zero for negative
    time, which is what makes the limits 0 and t rather than
    :math:`\pm\infty`.

    Parameters
    ----------
    x, h : array-like, shape (m,)
        Samples of the input and the impulse response on a uniform
        grid starting at t = 0.
    dt : float, default 1.0
        Sampling interval.

    Returns
    -------
    RichResult
        keys: ``y`` (m,), ``t`` (m,), ``dt``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3 (convolution for causal LTI systems).
    """
    x = np.asarray(x, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()
    if x.size != h.size:
        raise ValueError("x and h must be sampled on the same grid.")
    if x.size < 2:
        raise ValueError("need at least 2 samples.")
    dt = float(dt)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")

    m = x.size
    y = np.empty(m)
    for i in range(m):
        integrand = x[: i + 1] * h[i::-1]
        y[i] = np.trapezoid(integrand, dx=dt) if i > 0 else 0.0

    return RichResult(
        payload={
            "y": y,
            "t": np.arange(m) * dt,
            "dt": dt,
            "method": "Causal convolution integral y(t) = int_0^t x(tau) h(t-tau) dtau (trapezoid)",
        }
    )


def cheatsheet():
    return "rng032: y(t) = int_0^t x(tau) h(t-tau) dtau, trapezoid on the sample grid"
