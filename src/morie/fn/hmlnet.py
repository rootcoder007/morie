# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LeNet-5 CNN architecture."""

import numpy as np

from ._richresult import RichResult
from .grcos import geron_conv_output_size

__all__ = ["geron_lenet5"]

_METHOD = "LeNet-5 architecture resolution"


def geron_lenet5(n_classes=10, input_size=32, in_channels=1):
    """
    LeNet-5 CNN architecture.

    Formula: C1(6) -> S2 -> C3(16) -> S4 -> C5(120) -> F6(84) -> softmax(10)

    An architecture entry, so it resolves the architecture against a
    concrete input: every layer's output shape and parameter count for a
    ``32 x 32 x 1`` image.  The spatial arithmetic is delegated to
    :func:`morie.fn.grcos.geron_conv_output_size`.

    The 32x32 input is not an accident -- MNIST digits are 28x28 and are
    padded so that a 5x5 kernel with no padding leaves 28x28 after C1,
    and two 2x2 poolings then land exactly on 5x5 before C5 reduces it
    to 1x1.  Change ``input_size`` and the chain either stops dividing
    evenly or C5 stops being a full 1x1 convolution; that is checked
    rather than assumed.

    Pooling layers are counted as parameter-free (the modern
    convention); the original 1998 paper gave each pooling map a
    trainable coefficient and bias, and C3 was only partially connected,
    for 1516 parameters instead of 2416.  The convention in use is
    stated in the payload so the total is never a mystery number.

    Parameters
    ----------
    n_classes : int
        Output units.
    input_size : int
        Spatial size of the (square) input.
    in_channels : int
        Input channels.

    Returns
    -------
    result : RichResult
        Keys: layers, total_parameters, output_shape, receptive_field,
        estimate, n, method.

    Examples
    --------
    The classic configuration has 61,706 parameters:

    >>> r = geron_lenet5(n_classes=10)
    >>> r["total_parameters"]
    61706

    Layer by layer: C1 ``5*5*1*6 + 6 = 156``, C3 ``5*5*6*16 + 16 =
    2416``, C5 ``5*5*16*120 + 120 = 48120``, F6 ``120*84 + 84 = 10164``,
    output ``84*10 + 10 = 850``:

    >>> [(L["name"], L["parameters"]) for L in r["layers"] if L["parameters"]]
    [('C1', 156), ('C3', 2416), ('C5', 48120), ('F6', 10164), ('output', 850)]

    The spatial chain is 32 -> 28 -> 14 -> 10 -> 5 -> 1:

    >>> [L["output_shape"][0] for L in r["layers"][:5]]
    [28, 14, 10, 5, 1]

    Only the output layer changes with the class count:

    >>> geron_lenet5(n_classes=2)["total_parameters"]
    61026

    An input size that does not survive the chain is refused:

    >>> geron_lenet5(input_size=20)
    Traceback (most recent call last):
        ...
    ValueError: geron_lenet5: an input of 20 leaves 2x2 maps before C5 is reached; LeNet-5 needs a 32x32 input

    References
    ----------
    Géron Ch 12
    """
    k = int(n_classes)
    if k < 2:
        raise ValueError(f"geron_lenet5: n_classes must be at least 2, got {n_classes!r}")
    s = int(input_size)
    if s < 8:
        raise ValueError(f"geron_lenet5: input_size must be at least 8, got {input_size!r}")
    c_in = int(in_channels)
    if c_in < 1:
        raise ValueError(f"geron_lenet5: in_channels must be at least 1, got {in_channels!r}")

    layers = []
    total = 0

    def _conv(name, size, cin, cout, kernel):
        nonlocal total
        out = int(geron_conv_output_size(size, kernel, padding=0, stride=1)["out_size"][0])
        if out < 1:
            raise ValueError(
                f"geron_lenet5: a {kernel}x{kernel} kernel does not fit a {size}x{size} map at layer {name}"
            )
        p = kernel * kernel * cin * cout + cout
        total += p
        layers.append({"name": name, "type": "conv", "output_shape": (out, out, cout), "parameters": p})
        return out, cout

    def _pool(name, size, chan):
        out = int(geron_conv_output_size(size, 2, padding=0, stride=2)["out_size"][0])
        layers.append({"name": name, "type": "pool", "output_shape": (out, out, chan), "parameters": 0})
        return out, chan

    sz, ch = _conv("C1", s, c_in, 6, 5)
    sz, ch = _pool("S2", sz, ch)
    sz, ch = _conv("C3", sz, ch, 16, 5)
    sz, ch = _pool("S4", sz, ch)
    if sz < 5:
        raise ValueError(
            f"geron_lenet5: an input of {s} leaves {sz}x{sz} maps before C5 is reached; "
            f"LeNet-5 needs a 32x32 input"
        )
    sz, ch = _conv("C5", sz, ch, 120, 5)
    if sz != 1:
        raise ValueError(
            f"geron_lenet5: C5 produced a {sz}x{sz} map instead of 1x1; LeNet-5's chain only lands "
            f"exactly on 1x1 for a 32x32 input"
        )

    p_f6 = 120 * 84 + 84
    total += p_f6
    layers.append({"name": "F6", "type": "dense", "output_shape": (84,), "parameters": p_f6})
    p_out = 84 * k + k
    total += p_out
    layers.append({"name": "output", "type": "dense", "output_shape": (k,), "parameters": p_out})

    # Receptive field of a C5 unit back in the input: 5 -> 6 -> 10 -> 12 -> 16 ... computed by
    # walking the chain backwards with r <- (r - 1) * stride + kernel.
    rf = 1
    for kernel, stride in ((5, 1), (2, 2), (5, 1), (2, 2), (5, 1))[::-1]:
        rf = (rf - 1) * stride + kernel
    # each pooling doubles the jump, so account for the accumulated stride
    rf = 32 if rf > 32 else rf

    return RichResult(
        title="LeNet-5",
        summary_lines=[
            ("Input", f"{s} x {s} x {c_in}"),
            ("Total parameters", total),
            ("Classes", k),
        ],
        tables=[
            {
                "title": "Layers",
                "headers": ["layer", "type", "output", "params"],
                "rows": [[L["name"], L["type"], str(L["output_shape"]), L["parameters"]] for L in layers],
            }
        ],
        interpretation=(
            "Pooling counted as parameter-free and C3 fully connected: 61,706 parameters. The 1998 "
            "paper's trainable pooling and partial C3 connectivity give a different, smaller total."
        ),
        payload={
            "layers": layers,
            "total_parameters": int(total),
            "output_shape": (k,),
            "receptive_field": int(rf),
            "convention": "parameter-free pooling, fully-connected C3",
            "estimate": float(total),
            "n": len(layers),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlnet: LeNet-5 resolved against a 32x32 input -- per-layer shapes and 61,706 parameters"
