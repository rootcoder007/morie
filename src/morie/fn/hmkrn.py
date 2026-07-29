# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convolutional filter (kernel): learnable weight tensor."""

import numpy as np

from ._richresult import RichResult
from .hmhei import geron_he_init

__all__ = ["geron_filter_kernel"]

_METHOD = "Convolutional filter tensor"


def geron_filter_kernel(kh, kw, c_in, c_out, seed=0, init="he"):
    """
    Convolutional filter (kernel): learnable weight tensor.

    Formula: K in R^(kh x kw x C_in x C_out)

    A convolutional layer's parameters do not depend on the input's
    spatial size at all -- only on the kernel and the channel counts.
    That is the whole reason convolution scales: a 3x3x64x64 filter bank
    is 36,928 parameters whether the image is 32x32 or 4096x4096, where a
    dense layer between the same tensors would be billions.

    The fan-in for initialization is ``kh * kw * c_in`` -- every weight
    that contributes to one output unit -- not ``c_in``.  Using the
    channel count alone inflates the variance by the kernel area, which
    is the classic conv-init mistake; the variance is delegated to
    :func:`morie.fn.hmhei.geron_he_init` with the correct fan-in.

    Parameters
    ----------
    kh, kw : int
        Kernel height and width (positive).
    c_in, c_out : int
        Input and output channels (positive).
    seed : int
        Seed for the draw.
    init : {"he", "zeros"}
        Initialization scheme.

    Returns
    -------
    result : RichResult
        Keys: kernel, bias, shape, n_parameters, fan_in, std,
        estimate, n, method.

    Examples
    --------
    A 3x3 filter over 64 input and 64 output channels:

    >>> r = geron_filter_kernel(3, 3, 64, 64, seed=0)
    >>> r["shape"]
    (3, 3, 64, 64)
    >>> r["n_parameters"]
    36928

    The fan-in counts the whole receptive volume, so the target sd is
    ``sqrt(2 / (3*3*64)) = sqrt(2/576)``:

    >>> r["fan_in"]
    576
    >>> round(r["std"], 9) == round(float(np.sqrt(2 / 576)), 9)
    True

    Weights plus biases: ``3*3*64*64 = 36864`` weights and 64 biases:

    >>> int(r["kernel"].size), int(r["bias"].size)
    (36864, 64)

    The count is independent of image size -- there is no image here at
    all, which is the point:

    >>> geron_filter_kernel(1, 1, 3, 2)["n_parameters"]
    8

    References
    ----------
    Géron Ch 12
    """
    h, w = int(kh), int(kw)
    ci, co = int(c_in), int(c_out)
    for name, v in (("kh", h), ("kw", w), ("c_in", ci), ("c_out", co)):
        if v < 1:
            raise ValueError(f"geron_filter_kernel: {name} must be a positive integer, got {v}")
    if init not in ("he", "zeros"):
        raise ValueError(f"geron_filter_kernel: init must be 'he' or 'zeros', got {init!r}")

    fan_in = h * w * ci
    if init == "zeros":
        K = np.zeros((h, w, ci, co))
        std = 0.0
    else:
        flat = geron_he_init(fan_in, seed=int(seed), fan_out=co)
        std = float(flat["std_target"])
        K = np.asarray(flat["W"]).reshape(h, w, ci, co)
    bias = np.zeros(co)
    n_params = int(K.size + bias.size)

    return RichResult(
        title="Convolutional filter",
        summary_lines=[
            ("Shape", f"{h} x {w} x {ci} x {co}"),
            ("Parameters", n_params),
            ("fan_in", fan_in),
            ("Target sd", std),
        ],
        interpretation=(
            "Parameter count is independent of the input's spatial size; the fan-in is the whole "
            "receptive volume kh*kw*c_in, not c_in."
        ),
        payload={
            "kernel": K,
            "bias": bias,
            "shape": (h, w, ci, co),
            "n_parameters": n_params,
            "fan_in": int(fan_in),
            "std": std,
            "estimate": float(n_params),
            "n": int(K.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmkrn: conv filter tensor (kh, kw, c_in, c_out), He-initialised on fan_in = kh*kw*c_in"
