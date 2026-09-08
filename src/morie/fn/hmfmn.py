# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FashionMNIST image classifier: CNN on 28x28 gray images."""

from . import _array_core as np

from ._richresult import RichResult
from .grcos import geron_conv_output_size

__all__ = ["geron_fashion_mnist"]

_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def geron_fashion_mnist(epochs=10, lr=0.001, batch_size=32, n_classes=10, input_size=28, filters=(32, 64)):
    """
    FashionMNIST image classifier: CNN on 28x28 gray images.

    Formula: CNN -> flatten -> FC -> softmax(10)

    An architecture specification, resolved against the concrete 28x28x1
    input in the ``hmalex`` manner. Each conv/pool output size is computed
    by :func:`morie.fn.grcos.geron_conv_output_size` -- delegated, so the
    ``floor((n - k + 2p)/s) + 1`` arithmetic lives in one place -- and the
    parameter counts follow from the resolved shapes.

    The number worth reading is where the parameters *are*: with two
    conv blocks on 28x28, the flatten feeds a dense layer holding the
    large majority of the model, which is why global pooling or a smaller
    dense layer is the first thing to try when a small CNN overfits.
    ``fc_share`` reports that fraction.

    ``epochs``, ``lr`` and ``batch_size`` are validated and returned as
    the training configuration together with ``steps_per_epoch`` for the
    60000-image training set; no training is performed (see ``hmclsn``
    for a trained classifier).

    Parameters
    ----------
    epochs : int, default 10
    lr : float, default 0.001
    batch_size : int, default 32
    n_classes : int, default 10
    input_size : int, default 28
    filters : sequence of int, default (32, 64)
        Channels of each conv block (conv 3x3 then 2x2 max-pool).

    Returns
    -------
    result : RichResult
        Keys: layers, total_params, flatten_dim, fc_share, class_names,
        output_shape, training_config, steps_per_epoch, estimate, n,
        method.

    Examples
    --------
    The default two-block CNN on 28x28, resolved exactly: 3x3 valid
    convolutions give 26 and 11, each halved by pooling.

    >>> r = geron_fashion_mnist()
    >>> [l["out"] for l in r["layers"] if l["kind"] in ("conv", "pool")]
    [26, 13, 11, 5]
    >>> r["flatten_dim"]
    1600
    >>> r["total_params"]
    225034

    Most of the model is the dense layer after the flatten:

    >>> round(r["fc_share"], 3)
    0.916
    >>> r["class_names"][0], len(r["class_names"])
    ('T-shirt/top', 10)

    The training configuration is validated and reported:

    >>> r["steps_per_epoch"]
    1875
    >>> r["training_config"]["lr"]
    0.001

    A negative learning rate is rejected:

    >>> geron_fashion_mnist(lr=-0.1)
    Traceback (most recent call last):
      ...
    ValueError: geron_fashion_mnist: lr must be positive and finite, got -0.1

    References
    ----------
    Géron Ch 10
    """
    E = int(epochs)
    if E < 1:
        raise ValueError(f"geron_fashion_mnist: epochs must be >= 1, got {epochs!r}")
    eta = float(lr)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"geron_fashion_mnist: lr must be positive and finite, got {lr!r}")
    bs = int(batch_size)
    if bs < 1:
        raise ValueError(f"geron_fashion_mnist: batch_size must be >= 1, got {batch_size!r}")
    C = int(n_classes)
    if C < 2:
        raise ValueError(f"geron_fashion_mnist: n_classes must be >= 2, got {n_classes!r}")
    S = int(input_size)
    if S < 4:
        raise ValueError(f"geron_fashion_mnist: input_size must be >= 4, got {input_size!r}")
    chans = [int(f) for f in filters]
    if not chans or any(f < 1 for f in chans):
        raise ValueError(f"geron_fashion_mnist: filters must be positive, got {filters!r}")

    layers = []
    size = S
    ch = 1
    for f in chans:
        out = int(np.ravel(geron_conv_output_size(size, kernel=3, padding=0, stride=1)["out_size"])[0])
        if out < 1:
            raise ValueError(f"geron_fashion_mnist: input_size={S} is too small for {len(chans)} conv blocks")
        layers.append({"kind": "conv", "kernel": 3, "channels": f, "out": out, "params": f * (3 * 3 * ch) + f})
        size, ch = out, f
        out = int(np.ravel(geron_conv_output_size(size, kernel=2, padding=0, stride=2)["out_size"])[0])
        if out < 1:
            raise ValueError(f"geron_fashion_mnist: input_size={S} is too small for {len(chans)} conv blocks")
        layers.append({"kind": "pool", "kernel": 2, "channels": ch, "out": out, "params": 0})
        size = out

    flat = int(size * size * ch)
    dense = 128
    layers.append({"kind": "flatten", "out": flat, "channels": ch, "params": 0})
    layers.append({"kind": "fc", "out": dense, "channels": dense, "params": flat * dense + dense})
    layers.append({"kind": "fc", "out": C, "channels": C, "params": dense * C + C, "activation": "softmax"})

    total = int(sum(l["params"] for l in layers))
    fc_params = int(sum(l["params"] for l in layers if l["kind"] == "fc"))

    return RichResult(
        title="FashionMNIST CNN",
        summary_lines=[("Total parameters", total), ("Flatten dim", flat), ("Classes", C)],
        tables=[{"title": "Layers", "headers": ["kind", "out", "channels", "params"],
                 "rows": [[l["kind"], l["out"], l["channels"], l["params"]] for l in layers]}],
        interpretation="Nearly all the parameters sit in the dense layer after the flatten; that is where overfitting starts.",
        payload={
            "layers": layers,
            "total_params": total,
            "conv_params": int(sum(l["params"] for l in layers if l["kind"] == "conv")),
            "fc_params": fc_params,
            "fc_share": float(fc_params / total),
            "flatten_dim": flat,
            "class_names": list(_CLASSES[:C]) if C <= len(_CLASSES) else [f"class_{i}" for i in range(C)],
            "output_shape": (C,),
            "training_config": {"epochs": E, "lr": eta, "batch_size": bs, "loss": "sparse categorical cross-entropy"},
            "steps_per_epoch": int(np.ceil(60000 / bs)),
            "estimate": float(total),
            "n": int(len(layers)),
            "method": "FashionMNIST CNN resolved to concrete shapes; output sizes delegated to grcos",
        },
    )


def cheatsheet():
    return "hmfmn: FashionMNIST image classifier: CNN on 28x28 gray images"
