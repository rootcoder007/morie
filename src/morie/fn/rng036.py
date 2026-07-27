# morie.fn -- function file (rootcoder007/morie)
"""Discrete-time causal convolution sum."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_discrete_convolution_causal"]


def rangayyan_ch3_discrete_convolution_causal(x, h, n=None):
    r"""Causal discrete convolution :math:`y(n) = \sum_{k=0}^{n} x(k) h(n-k)`.

    The sum runs only over past and present inputs, so ``y[n]`` never
    depends on ``x[n+1]`` and beyond -- the definition of a causal
    system.

    Parameters
    ----------
    x, h : array-like
        Input signal and impulse response, both taken as zero for
        negative indices.
    n : int, optional
        Return only ``y(n)`` (a float) instead of the whole sequence.

    Returns
    -------
    RichResult
        keys: ``y`` (full sequence of length len(x) + len(h) - 1),
        ``value`` (y(n) when n given, else None), ``n``, ``method``.

    References
    ----------
    Rangayyan, R. M. (2024). *Biomedical Signal Analysis* (3rd ed.).
    Wiley-IEEE Press. Ch. 3 (linear time-invariant systems and the
    causal convolution sum).
    """
    x = np.asarray(x, dtype=float).ravel()
    h = np.asarray(h, dtype=float).ravel()
    if x.size == 0 or h.size == 0:
        raise ValueError("x and h must be non-empty.")
    y = np.convolve(x, h)

    value = None
    if n is not None:
        n = int(n)
        if not 0 <= n < y.size:
            raise ValueError(f"n must lie in [0, {y.size - 1}], got {n}.")
        value = float(y[n])

    return RichResult(
        payload={
            "y": y,
            "value": value,
            "n": n,
            "method": "Causal discrete convolution sum y(n) = sum_k x(k) h(n-k)",
        }
    )


def cheatsheet():
    return "rng036: y(n) = sum_{k=0}^{n} x(k) h(n-k)"
