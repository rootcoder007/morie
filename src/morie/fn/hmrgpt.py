# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression MLP expressed as a PyTorch nn.Sequential architecture."""

from . import _array_core as np

from ._richresult import RichResult
from .hmregn import geron_regression_mlp

__all__ = ["geron_regression_mlp_pytorch"]


def geron_regression_mlp_pytorch(X, y, hidden=(8,), epochs=400, lr=0.05, seed=0):
    """
    Regression MLP with a PyTorch ``nn.Sequential``.

    Formula: nn.Linear -> ReLU -> nn.Linear ... -> nn.Linear(1)

    morie.fn is numpy-only, so NO torch call is made here and none is
    faked. What the function does instead is the part that is actually
    checkable: it RESOLVES the architecture against the concrete input --
    every layer's in/out features and parameter count, in
    ``nn.Sequential`` order, verified against the data you passed -- and
    trains that exact network by DELEGATING to
    :func:`~morie.fn.hmregn.geron_regression_mlp`, whose maths is the
    same forward pass and the same MSE loss torch would compute.

    ``layers`` is the module list you would hand to ``nn.Sequential``,
    and ``n_parameters`` matches what ``sum(p.numel() for p in
    model.parameters())`` would report for it. The one thing this cannot
    reproduce is torch's own initialisation stream, so weights differ
    from a torch run with the same seed even though the architecture and
    objective are identical.

    Parameters
    ----------
    X : array-like, shape (m, n)
    y : array-like, shape (m,) or (m, k)
    hidden : sequence of int, default (8,)
        Hidden widths.
    epochs : int, default 400
    lr : float, default 0.05
    seed : int, default 0

    Returns
    -------
    result : RichResult
        Keys: layers, n_parameters, predict, predictions, mse,
        loss_history, estimate, n, method.

    Examples
    --------
    >>> X = [[1.0], [2.0], [3.0], [4.0]]
    >>> r = geron_regression_mlp_pytorch(X, [2.0, 4.0, 6.0, 8.0], hidden=(8,), epochs=800, lr=0.02)
    >>> r["layers"]
    ['Linear(in_features=1, out_features=8)', 'ReLU()', 'Linear(in_features=8, out_features=1)']
    >>> int(r["n_parameters"])
    25
    >>> bool(r["mse"] < 0.05)
    True

    Two hidden layers add the second Linear/ReLU pair:

    >>> geron_regression_mlp_pytorch(X, [2.0, 4.0, 6.0, 8.0], hidden=(4, 4), epochs=10)["layers"]
    ['Linear(in_features=1, out_features=4)', 'ReLU()', 'Linear(in_features=4, out_features=4)', 'ReLU()', 'Linear(in_features=4, out_features=1)']

    References
    ----------
    Geron Ch 10
    """
    base = geron_regression_mlp(X, y, hidden_sizes=hidden, epochs=epochs, lr=lr, seed=seed)
    sizes = list(base["sizes"])
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(f"Linear(in_features={sizes[i]}, out_features={sizes[i + 1]})")
        if i < len(sizes) - 2:
            layers.append("ReLU()")

    return RichResult(
        title="Regression MLP (nn.Sequential architecture)",
        summary_lines=[("Modules", len(layers)), ("Parameters", int(base["n_parameters"])), ("Training MSE", float(base["mse"]))],
        warnings=[
            "No torch call is made: morie.fn is numpy-only. The architecture, the forward pass and the MSE "
            "objective are the ones torch would use; the initialisation stream is not."
        ],
        interpretation="The shapes are resolved against your data, so a mismatch surfaces here rather than at run time.",
        payload={
            "layers": layers,
            "sizes": sizes,
            "n_parameters": int(base["n_parameters"]),
            "predict": base["predict"],
            "predictions": base["predictions"],
            "mse": float(base["mse"]),
            "loss_history": base["loss_history"],
            "weights": base["weights"],
            "biases": base["biases"],
            "uses_torch": False,
            "estimate": base["predictions"],
            "n": int(base["n"]),
            "method": "nn.Sequential architecture resolved on the data; training delegated to morie.fn.hmregn",
        },
    )


def cheatsheet():
    return "hmrgpt: Regression MLP as an nn.Sequential architecture (numpy-trained)"
