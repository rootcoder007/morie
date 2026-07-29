# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neurons-per-layer heuristic for a fully connected network."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_neurons_per_layer"]


def geron_neurons_per_layer(n_features, n_layers=1, n_outputs=1, width=None):
    """
    Neurons-per-layer heuristic: similar width across hidden layers.

    Formula: width chosen in range [d, 2d] for d features

    The old advice was a funnel, wide at the bottom and narrow at the
    top; the practice Geron reports is a CONSTANT width, which performs
    as well or better and leaves one hyperparameter instead of many. The
    width is not the thing to tune first anyway: pick something ample,
    then control capacity with early stopping and regularisation. The
    parameter count that width implies is returned, since that -- not the
    neuron count -- is what has to fit in memory and in the data.

    Parameters
    ----------
    n_features : int
        Input dimension d (>= 1).
    n_layers : int, default 1
        Number of hidden layers (>= 1).
    n_outputs : int, default 1
        Output dimension (>= 1).
    width : int, optional
        Use this width instead of the heuristic.

    Returns
    -------
    result : RichResult
        Keys: width, width_range, n_parameters, parameters_per_layer,
        estimate, n, method.

    Examples
    --------
    >>> r = geron_neurons_per_layer(10, n_layers=2)
    >>> r["width_range"], int(r["width"])
    ((10, 20), 20)

    Parameters: (10*20 + 20) + (20*20 + 20) + (20*1 + 1) = 661

    >>> int(r["n_parameters"])
    661
    >>> [int(v) for v in r["parameters_per_layer"]]
    [220, 420, 21]

    References
    ----------
    Geron Ch 9
    """
    d = int(n_features)
    L = int(n_layers)
    k = int(n_outputs)
    if d < 1:
        raise ValueError(f"geron_neurons_per_layer: n_features must be >= 1, got {n_features!r}")
    if L < 1:
        raise ValueError(f"geron_neurons_per_layer: n_layers must be >= 1, got {n_layers!r}")
    if k < 1:
        raise ValueError(f"geron_neurons_per_layer: n_outputs must be >= 1, got {n_outputs!r}")
    if width is None:
        w = 2 * d
    else:
        w = int(width)
        if w < 1:
            raise ValueError(f"geron_neurons_per_layer: width must be >= 1, got {width!r}")

    per = [d * w + w] + [w * w + w] * (L - 1) + [w * k + k]
    total = int(sum(per))
    return RichResult(
        title="Neurons per layer",
        summary_lines=[("Width", w), ("Hidden layers", L), ("Parameters", total)],
        interpretation="Constant width beats a funnel and costs one hyperparameter; cap capacity by early stopping.",
        payload={
            "width": int(w),
            "width_range": (d, 2 * d),
            "n_layers": L,
            "n_parameters": total,
            "parameters_per_layer": [int(v) for v in per],
            "estimate": int(w),
            "n": int(d),
            "method": "Constant-width hidden-layer heuristic with parameter count",
        },
    )


def cheatsheet():
    return "hmnpl: Neurons-per-layer heuristic (constant width in [d, 2d])"
