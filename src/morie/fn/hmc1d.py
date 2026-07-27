# morie.fn -- function file (rootcoder007/morie)
"""Causal 1D convolution: output at time t depends only on t' <= t."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_causal_1d_conv"]


def geron_causal_1d_conv(x, kernel):
    r"""Left-padded ("causal") 1-D convolution.

    .. math:: y_t = \sum_{k=0}^{K-1} w_k\, x_{t-k},

    with x taken as zero before the start. Equivalent to zero-padding
    the input by K-1 on the left and running a valid convolution, which
    is how a causal Conv1D layer avoids seeing the future -- the
    property WaveNet-style forecasting stacks rely on.

    Parameters
    ----------
    x : array-like, shape (n,)
        Input sequence.
    kernel : array-like, shape (K,)
        Filter taps, ``kernel[0]`` multiplying the current sample.

    Returns
    -------
    RichResult
        keys: ``y`` (n,), ``kernel_size``, ``n``, ``method``.

    References
    ----------
    Geron, A. (2022). *Hands-On Machine Learning with Scikit-Learn,
    Keras, and TensorFlow* (3rd ed.). O'Reilly. Ch. 15 (processing
    sequences: causal / "causal padding" 1D convolutions and WaveNet).
    """
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(kernel, dtype=float).ravel()
    if x.size == 0 or w.size == 0:
        raise ValueError("x and kernel must be non-empty.")
    if w.size > x.size:
        raise ValueError(f"kernel of length {w.size} longer than the input ({x.size}).")

    padded = np.concatenate([np.zeros(w.size - 1), x])
    y = np.convolve(padded, w, mode="valid")

    return RichResult(
        payload={
            "y": y,
            "kernel_size": int(w.size),
            "n": int(x.size),
            "method": "Causal 1D convolution (left zero-padding, no future leakage)",
        }
    )


def cheatsheet():
    return "hmc1d: y_t = sum_k w_k x_{t-k} via left zero-padding"
