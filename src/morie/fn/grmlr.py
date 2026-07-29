# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression MLP output head."""

import numpy as np

from ._richresult import RichResult
from .grlinf import geron_linear_layer_forward

__all__ = ["geron_regression_mlp_output"]

_METHOD = "Regression MLP output head"

_ACTS = ("identity", "softplus", "relu", "sigmoid")


def geron_regression_mlp_output(a_last, W_out, b_out, activation="identity"):
    r"""Final layer of a regression MLP.

    .. math::
        \hat y = W_{\text{out}}\, a_{L-1} + b_{\text{out}}

    with, by default, no nonlinearity: a bounded output activation would
    cap what the network can predict.  ``softplus`` is offered because
    Géron recommends it when the target is known to be positive, and
    ``sigmoid`` when it is known to lie in ``(0, 1)`` -- those are the
    only reasons to leave ``identity``.

    The affine part is delegated to
    :func:`morie.fn.grlinf.geron_linear_layer_forward`.

    Parameters
    ----------
    a_last : array-like, shape (h,) or (m, h)
        Activations of the last hidden layer.
    W_out : array-like, shape (out, h)
    b_out : array-like, shape (out,) or scalar
    activation : {"identity", "softplus", "relu", "sigmoid"}, optional

    Returns
    -------
    RichResult
        Payload keys ``prediction``, ``preactivation``, ``activation``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 9, Regression MLPs section.

    Examples
    --------
    Identity output passes the affine value straight through:

    >>> r = geron_regression_mlp_output([1.0, 2.0], [[1.0, 1.0]], [0.5])
    >>> r["prediction"]
    [3.5]

    Softplus of 0 is ``log 2``, and it is strictly positive -- which is
    the whole point of using it:

    >>> r2 = geron_regression_mlp_output([1.0], [[1.0]], [-1.0], activation="softplus")
    >>> round(r2["prediction"][0], 10)
    0.6931471806
    """
    if activation not in _ACTS:
        raise ValueError(f"activation must be one of {_ACTS}, got {activation!r}.")
    inner = geron_linear_layer_forward(a_last, W_out, b_out)
    Z = np.asarray(inner["output"], dtype=float)

    if activation == "identity":
        Y = Z
    elif activation == "softplus":
        Y = np.logaddexp(0.0, Z)
    elif activation == "relu":
        Y = np.maximum(Z, 0.0)
    else:
        Y = 1.0 / (1.0 + np.exp(-Z))

    return RichResult(
        title="Regression MLP output",
        summary_lines=[("Activation", activation), ("Outputs", inner["out_features"])],
        payload={
            "prediction": Y.tolist(),
            "preactivation": Z.tolist(),
            "activation": activation,
            "out_features": inner["out_features"],
            "estimate": Y.tolist(),
            "n": inner["n"],
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grmlr: y_hat = W_out a + b_out, identity output by default (delegates to grlinf)"
