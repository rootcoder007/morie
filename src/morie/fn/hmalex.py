# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AlexNet: deep CNN for ImageNet with ReLU and dropout."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_alexnet"]

# (kind, filters/units, kernel, stride, padding) -- the 2012 architecture.
_SPEC = [
    ("conv", 96, 11, 4, 0),
    ("pool", None, 3, 2, 0),
    ("conv", 256, 5, 1, 2),
    ("pool", None, 3, 2, 0),
    ("conv", 384, 3, 1, 1),
    ("conv", 384, 3, 1, 1),
    ("conv", 256, 3, 1, 1),
    ("pool", None, 3, 2, 0),
    ("fc", 4096, None, None, None),
    ("fc", 4096, None, None, None),
]


def geron_alexnet(n_classes=1000, input_size=227, in_channels=3, dropout=0.5):
    """
    AlexNet: deep CNN for ImageNet with ReLU and dropout.

    Formula: 5 conv -> 3 FC; ReLU; dropout 0.5; 60M params

    Builds the architecture and resolves it against a concrete input size:
    every layer's output shape and parameter count is computed from
    ``out = floor((in - k + 2p)/s) + 1`` and
    ``params = filters * (k*k*in_channels) + filters``, so an input size that
    collapses a feature map to zero is reported as an error rather than
    silently producing a negative dimension.

    Parameters
    ----------
    n_classes : int
        Output units of the final fully-connected layer (>= 1).
    input_size : int
        Side length of the (square) input image.
    in_channels : int
        Input channels (3 for RGB).
    dropout : float
        Dropout rate applied to the two hidden FC layers; in [0, 1).

    Returns
    -------
    result : RichResult
        Keys: layers, total_params, trainable_params, output_shape,
        flatten_dim, estimate, n, method.

    Examples
    --------
    >>> r = geron_alexnet(1000)
    >>> r["total_params"]
    62378344
    >>> r["flatten_dim"]
    9216
    >>> [l["out"] for l in r["layers"] if l["kind"] == "conv"]
    [55, 27, 13, 13, 13]
    >>> geron_alexnet(10)["total_params"] - geron_alexnet(1000)["total_params"]
    -4056030

    References
    ----------
    Géron Ch 12
    """
    C = int(n_classes)
    if C < 1:
        raise ValueError("geron_alexnet: n_classes must be >= 1")
    S = int(input_size)
    if S < 1:
        raise ValueError("geron_alexnet: input_size must be >= 1")
    ch = int(in_channels)
    if ch < 1:
        raise ValueError("geron_alexnet: in_channels must be >= 1")
    p_drop = float(dropout)
    if not (0.0 <= p_drop < 1.0):
        raise ValueError(f"geron_alexnet: dropout must lie in [0, 1), got {p_drop}")

    layers = []
    size = S
    channels = ch
    flatten_dim = None
    for kind, units, k, s, pad in _SPEC:
        if kind in ("conv", "pool"):
            out = (size - k + 2 * pad) // s + 1
            if out < 1:
                raise ValueError(
                    f"geron_alexnet: input_size={S} is too small; a {kind} layer with kernel {k} "
                    f"stride {s} would produce a {out}x{out} feature map"
                )
            if kind == "conv":
                params = units * (k * k * channels) + units
                channels = units
            else:
                params = 0
            layers.append(
                {"kind": kind, "filters": units, "kernel": k, "stride": s, "pad": pad, "out": out, "params": params}
            )
            size = out
        else:
            if flatten_dim is None:
                flatten_dim = size * size * channels
                in_units = flatten_dim
            else:
                in_units = layers[-1]["filters"]
            params = in_units * units + units
            layers.append({"kind": "fc", "filters": units, "in": in_units, "out": units, "params": params, "dropout": p_drop})
    last_hidden = layers[-1]["filters"]
    layers.append({"kind": "fc", "filters": C, "in": last_hidden, "out": C, "params": last_hidden * C + C, "dropout": 0.0})

    total = int(sum(l["params"] for l in layers))

    return RichResult(
        title="AlexNet architecture",
        summary_lines=[("Total parameters", total), ("Flatten dim", int(flatten_dim)), ("Classes", C)],
        tables=[
            {
                "title": "Layers",
                "headers": ["#", "kind", "units", "out", "params"],
                "rows": [[i, l["kind"], l["filters"], l["out"], l["params"]] for i, l in enumerate(layers)],
            }
        ],
        payload={
            "layers": layers,
            "total_params": total,
            "trainable_params": total,
            "output_shape": (C,),
            "flatten_dim": int(flatten_dim),
            "conv_params": int(sum(l["params"] for l in layers if l["kind"] == "conv")),
            "fc_params": int(sum(l["params"] for l in layers if l["kind"] == "fc")),
            "dropout": p_drop,
            "estimate": float(total),
            "n": int(len(layers)),
            "method": "AlexNet architecture resolved to concrete shapes and parameter counts",
        },
    )


def cheatsheet():
    return "hmalex: AlexNet: deep CNN for ImageNet with ReLU and dropout"


# compact alias per ledger/NAMING.md
geronalexnet = geron_alexnet
