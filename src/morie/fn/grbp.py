# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Backpropagation: gradient of the loss w.r.t. each weight layer via the chain rule."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_backpropagation_gradient"]

_METHOD = "Backpropagation through a feed-forward net"

_ACTS = ("sigmoid", "tanh", "relu", "identity")


def _act_deriv(name, a):
    """phi'(z) expressed in terms of the post-activation value a."""
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "tanh":
        return 1.0 - a * a
    if name == "relu":
        return (a > 0).astype(float)
    return np.ones_like(a)


def geron_backpropagation_gradient(activations, weights, y_true,
                                   activation="sigmoid", output_activation=None):
    r"""Backpropagate a squared-error loss through a stack of dense layers.

    .. math::
        \delta_L = \nabla_a L(a_L) \odot \varphi'(z_L), \qquad
        \delta_l = (W_{l+1}^{\top}\delta_{l+1}) \odot \varphi'(z_l),
        \qquad \nabla_{W_l} L = a_{l-1}^{\top}\delta_l

    The loss is :math:`L = \tfrac12 \sum_i \|a_L^{(i)} - y^{(i)}\|^2`
    summed over the batch, so :math:`\nabla_a L = a_L - y`.  Derivatives
    are computed from the stored *post*-activation values, which is why
    only the activations -- not the pre-activations -- need to be kept.

    Parameters
    ----------
    activations : sequence of array-like
        ``[a_0, a_1, ..., a_L]``, each ``(m, n_l)``; ``a_0`` is the input
        batch. Must be one longer than ``weights``.
    weights : sequence of array-like
        ``[W_1, ..., W_L]``, ``W_l`` of shape ``(n_{l-1}, n_l)``.
    y_true : array-like, shape (m, n_L)
        Targets.
    activation : {"sigmoid", "tanh", "relu", "identity"}, optional
        Hidden-layer nonlinearity, default ``"sigmoid"``.
    output_activation : same choices, optional
        Output-layer nonlinearity; defaults to ``activation``.

    Returns
    -------
    RichResult
        Payload keys ``grad_weights`` (one per layer, input side first),
        ``grad_biases``, ``deltas``, ``loss``, ``grad_norm``,
        ``estimate`` (the loss), ``n``, ``method``.

    References
    ----------
    Géron Ch 9 / Ch 10, Backpropagation Algorithm section.

    Examples
    --------
    A single linear layer ``a1 = a0 @ W1`` with ``a0 = [1, 2]``,
    ``W1 = [[1], [1]]`` gives ``a1 = 3``; against ``y = 1`` the error is
    2, so ``dL/dW1 = a0^T * 2 = [[2], [4]]``:

    >>> r = geron_backpropagation_gradient(
    ...     [[[1.0, 2.0]], [[3.0]]], [[[1.0], [1.0]]], [[1.0]],
    ...     activation="identity")
    >>> r["grad_weights"][0]
    [[2.0], [4.0]]
    >>> round(r["loss"], 6)
    2.0

    With a sigmoid output the same error is damped by
    ``a(1-a) = 3*(1-3) = -6``:

    >>> r2 = geron_backpropagation_gradient(
    ...     [[[1.0, 2.0]], [[3.0]]], [[[1.0], [1.0]]], [[1.0]],
    ...     activation="sigmoid")
    >>> r2["deltas"][-1]
    [[-12.0]]
    """
    acts = [np.atleast_2d(np.asarray(a, dtype=float)) for a in activations]
    Ws = [np.atleast_2d(np.asarray(W, dtype=float)) for W in weights]
    y = np.atleast_2d(np.asarray(y_true, dtype=float))
    if len(acts) != len(Ws) + 1:
        raise ValueError(
            f"activations must be one longer than weights (got {len(acts)} "
            f"activations and {len(Ws)} weight matrices)."
        )
    if not Ws:
        raise ValueError("weights is empty; nothing to differentiate.")
    if activation not in _ACTS:
        raise ValueError(f"activation must be one of {_ACTS}, got {activation!r}.")
    out_act = activation if output_activation is None else output_activation
    if out_act not in _ACTS:
        raise ValueError(f"output_activation must be one of {_ACTS}, got {out_act!r}.")
    m = acts[0].shape[0]
    for i, a in enumerate(acts):
        if a.shape[0] != m:
            raise ValueError(f"activations[{i}] has {a.shape[0]} rows, expected {m}.")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"activations[{i}] contains non-finite values.")
    for l, W in enumerate(Ws):
        if W.shape != (acts[l].shape[1], acts[l + 1].shape[1]):
            raise ValueError(
                f"weights[{l}] has shape {W.shape}, expected "
                f"{(acts[l].shape[1], acts[l + 1].shape[1])}."
            )
        if not np.all(np.isfinite(W)):
            raise ValueError(f"weights[{l}] contains non-finite values.")
    if y.shape != acts[-1].shape:
        raise ValueError(
            f"y_true shape {y.shape} must match the output activation shape "
            f"{acts[-1].shape}."
        )

    L = len(Ws)
    deltas = [None] * L
    err = acts[-1] - y
    deltas[L - 1] = err * _act_deriv(out_act, acts[-1])
    for l in range(L - 2, -1, -1):
        deltas[l] = (deltas[l + 1] @ Ws[l + 1].T) * _act_deriv(activation, acts[l + 1])

    grads = [acts[l].T @ deltas[l] for l in range(L)]
    gbias = [deltas[l].sum(axis=0) for l in range(L)]
    loss = float(0.5 * np.sum(err**2))
    gnorm = float(np.sqrt(sum(float(np.sum(g**2)) for g in grads)))

    return RichResult(
        title="Backpropagation",
        summary_lines=[("Loss", loss), ("Layers", L), ("‖grad‖", gnorm)],
        payload={
            "grad_weights": [g.tolist() for g in grads],
            "grad_biases": [g.tolist() for g in gbias],
            "deltas": [d.tolist() for d in deltas],
            "loss": loss,
            "grad_norm": gnorm,
            "activation": activation,
            "output_activation": out_act,
            "estimate": loss,
            "n": int(m),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbp: backprop -- delta_L = (a_L - y)*phi'(z_L); delta_l = (W delta)*phi'; grad = a^T delta"
