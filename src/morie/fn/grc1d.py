# morie.fn -- function file (rootcoder007/morie)
"""Causal (masked) 1D convolution for time-series forecasting."""

import numpy as np

from ._richresult import RichResult
from .hmc1d import geron_causal_1d_conv

__all__ = ["geron_causal_1d_cnn"]


def geron_causal_1d_cnn(x, w, dilation=1, strict=False):
    r"""Causal Conv1D with optional dilation and strict masking.

    Same left-padded convolution as
    :func:`morie.fn.hmc1d.geron_causal_1d_conv`, plus the two knobs a
    forecasting stack needs:

    - ``dilation`` d spaces the taps d samples apart,
      :math:`y_t = \sum_k w_k x_{t-kd}` -- stacking layers with
      d = 1, 2, 4, ... is the WaveNet receptive-field trick;
    - ``strict=True`` drops the k = 0 tap so :math:`y_t` uses only
      *strictly* past inputs, which is what one-step-ahead forecasting
      requires.

    Parameters
    ----------
    x : array-like, shape (n,)
        Input sequence.
    w : array-like, shape (K,)
        Filter taps.
    dilation : int, default 1
        Spacing between taps.
    strict : bool, default False
        Exclude the current sample.

    Returns
    -------
    RichResult
        keys: ``y`` (n,), ``receptive_field``, ``dilation``,
        ``strict``, ``n``, ``method``.

    References
    ----------
    Geron, A. (2022). *Hands-On Machine Learning with Scikit-Learn,
    Keras, and TensorFlow* (3rd ed.). O'Reilly. Ch. 15 (causal
    padding; dilated causal convolutions / WaveNet).
    """
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()
    d = int(dilation)
    if d < 1:
        raise ValueError(f"dilation must be at least 1, got {d}.")
    if x.size == 0 or w.size == 0:
        raise ValueError("x and w must be non-empty.")

    # expand taps to a dilated (and optionally shifted) kernel
    span = (w.size - 1) * d + 1 + (d if strict else 0)
    kern = np.zeros(span)
    offset = d if strict else 0
    kern[offset + d * np.arange(w.size)] = w
    if span > x.size:
        raise ValueError(f"receptive field {span} exceeds the input length {x.size}.")

    out = geron_causal_1d_conv(x, kern)
    return RichResult(
        payload={
            "y": out["y"],
            "receptive_field": int(span),
            "dilation": d,
            "strict": bool(strict),
            "n": int(x.size),
            "method": "Dilated causal 1D convolution (strict masking optional)",
        }
    )


def cheatsheet():
    return "grc1d: dilated causal conv y_t = sum_k w_k x_{t-kd}; strict drops the k=0 tap"
