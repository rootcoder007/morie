# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Xception: extreme inception using depthwise separable convolutions."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_xception", "separable_params"]


def separable_params(k, c_in, c_out):
    """Depthwise separable convolution weights: ``k*k*c_in + c_in*c_out``.

    A standard convolution costs ``k*k*c_in*c_out``, so the separable form
    saves a factor of ``1/c_out + 1/k^2``.
    """
    return int(k * k * c_in + c_in * c_out)


def geron_xception(n_classes=1000, in_channels=3, input_size=299):
    """
    Xception: extreme inception using depthwise separable convolutions.

    Formula: depthwise conv -> 1x1 pointwise conv -> residual

    Resolves Chollet's (2017) architecture -- entry flow, 8 repeats of the
    middle flow, exit flow -- into concrete layers, output shapes and
    parameter counts. Separable convolutions are counted as
    ``k*k*c_in`` depthwise weights plus ``c_in*c_out`` pointwise weights
    (see :func:`separable_params`); batch norm contributes ``2*C``
    trainable (scale, shift) and ``2*C`` non-trainable (moving mean,
    variance) values per layer. For the ImageNet configuration this
    reproduces the reference count of 22,855,952 trainable parameters.

    Parameters
    ----------
    n_classes : int, default 1000
        Units in the final fully-connected layer (>= 1).
    in_channels : int, default 3
        Input channels.
    input_size : int, default 299
        Side of the square input; used to resolve the spatial shapes.

    Returns
    -------
    result : RichResult
        Keys: layers, total_params, trainable_params, non_trainable_params,
        n_separable, standard_conv_params, savings_ratio, estimate, n, method.

    Examples
    --------
    >>> r = geron_xception(1000)
    >>> int(r["trainable_params"])
    22855952
    >>> int(r["total_params"])
    22910480
    >>> int(r["n_separable"])
    34

    A 3x3 separable conv from 728 to 728 channels costs far less than the
    dense equivalent:

    >>> separable_params(3, 728, 728)
    536536
    >>> round(536536 / (3 * 3 * 728 * 728), 4)
    0.1125

    References
    ----------
    Géron Ch 12
    """
    K = int(n_classes)
    if K < 1:
        raise ValueError(f"geron_xception: n_classes must be >= 1, got {K}")
    cin = int(in_channels)
    if cin < 1:
        raise ValueError(f"geron_xception: in_channels must be >= 1, got {cin}")
    size = int(input_size)
    if size < 32:
        raise ValueError(
            f"geron_xception: input_size {size} is too small; the entry flow downsamples by 32 "
            "and the feature map would collapse"
        )

    layers = []
    bn_channels = 0
    spatial = size

    def _out(k, s, p):
        return (spatial + 2 * p - k) // s + 1

    def add(kind, params, c_out, bn=True, note=""):
        nonlocal bn_channels
        if bn:
            bn_channels += c_out
        layers.append(
            {"kind": kind, "params": int(params), "channels": int(c_out), "out": int(spatial), "note": note}
        )

    # -- entry flow -------------------------------------------------------
    spatial = _out(3, 2, 0)
    add("conv3x3/s2", 3 * 3 * cin * 32, 32)
    c = 32
    add("conv3x3", 3 * 3 * c * 64, 64)
    c = 64
    for width in (128, 256, 728):
        add("separable3x3", separable_params(3, c, width), width)
        add("separable3x3", separable_params(3, width, width), width)
        spatial = (spatial - 3 + 2 * 1) // 2 + 1  # maxpool 3x3 stride 2, pad 1
        add("maxpool3x3/s2", 0, width, bn=False)
        add("conv1x1 shortcut/s2", c * width, width)
        c = width

    # -- middle flow (8 identical blocks) ---------------------------------
    for _ in range(8):
        for _ in range(3):
            add("separable3x3", separable_params(3, 728, 728), 728, note="middle flow")

    # -- exit flow --------------------------------------------------------
    add("separable3x3", separable_params(3, 728, 728), 728)
    add("separable3x3", separable_params(3, 728, 1024), 1024)
    spatial = (spatial - 3 + 2 * 1) // 2 + 1
    add("maxpool3x3/s2", 0, 1024, bn=False)
    add("conv1x1 shortcut/s2", 728 * 1024, 1024)
    add("separable3x3", separable_params(3, 1024, 1536), 1536)
    add("separable3x3", separable_params(3, 1536, 2048), 2048)
    add("global_avg_pool", 0, 2048, bn=False)
    layers.append({"kind": "fc", "params": 2048 * K + K, "channels": K, "out": 1, "note": "classifier"})

    if spatial < 1:
        raise ValueError(f"geron_xception: input_size {size} collapses the feature map before the classifier")

    weight_params = int(sum(l["params"] for l in layers))
    bn_trainable = 2 * bn_channels
    trainable = weight_params + bn_trainable
    non_trainable = 2 * bn_channels
    n_sep = sum(1 for l in layers if l["kind"] == "separable3x3")
    std_equiv = 0
    c_prev = cin
    for l in layers:
        if l["kind"] == "separable3x3":
            # what the same layer would cost as a dense 3x3 convolution
            c_in_l = c_prev
            std_equiv += 3 * 3 * c_in_l * l["channels"]
        if l["channels"] and l["kind"] != "fc":
            c_prev = l["channels"]
    sep_total = sum(l["params"] for l in layers if l["kind"] == "separable3x3")

    return RichResult(
        title="Xception architecture",
        summary_lines=[
            ("Trainable parameters", trainable),
            ("Total (incl. BN buffers)", trainable + non_trainable),
            ("Separable conv layers", n_sep),
            ("Final feature map", f"{spatial}x{spatial}x2048"),
        ],
        tables=[
            {
                "title": "Layers",
                "headers": ["#", "kind", "channels", "params"],
                "rows": [[i, l["kind"], l["channels"], l["params"]] for i, l in enumerate(layers)],
            }
        ],
        interpretation=(
            "Xception replaces Inception modules with depthwise separable convolutions: the same "
            "depth costs roughly 1/k^2 of the dense convolution weights, so the saved budget buys depth."
        ),
        payload={
            "layers": layers,
            "total_params": int(trainable + non_trainable),
            "trainable_params": int(trainable),
            "non_trainable_params": int(non_trainable),
            "weight_params": weight_params,
            "bn_channels": int(bn_channels),
            "n_separable": int(n_sep),
            "separable_params_total": int(sep_total),
            "standard_conv_params": int(std_equiv),
            "savings_ratio": float(sep_total / std_equiv),
            "output_shape": (K,),
            "final_map": (int(spatial), int(spatial), 2048),
            "estimate": float(trainable),
            "n": int(len(layers)),
            "method": "Xception resolved to concrete layers, shapes and parameter counts",
        },
    )


def cheatsheet():
    return "hmxcpt: Xception: extreme inception using depthwise separable convolutions"


# compact alias per ledger/NAMING.md
geronxception = geron_xception
