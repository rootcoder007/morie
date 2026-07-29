# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hebb's rule for the perceptron weight update."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_hebb_rule"]

_METHOD = "Perceptron learning rule (Hebb)"


def geron_hebb_rule(x, y_true, y_pred, w, eta):
    r"""The perceptron learning rule.

    .. math::
        w_{i,j} \leftarrow w_{i,j}
        + \eta\,(y_j - \hat y_j)\, x_i

    Two things follow directly from the form.  When the prediction is
    right the error is zero and *nothing moves* -- the perceptron only
    learns from its mistakes.  And the update is an outer product
    :math:`\eta\, x (y - \hat y)^{\mathsf T}`, so an input feature that
    is zero on this instance never has its weights touched.

    Predictions are the caller's: pass the thresholded output of
    :func:`morie.fn.grhev.geron_heaviside_step` for a classic TLU.

    Parameters
    ----------
    x : array-like, shape (n_in,)
        One instance.
    y_true, y_pred : array-like, shape (n_out,)
        Target and predicted output of each unit.
    w : array-like, shape (n_in, n_out)
        Current weights.
    eta : float
        Positive learning rate.

    Returns
    -------
    RichResult
        Payload keys ``w_new``, ``delta_w``, ``error``,
        ``converged`` (True when the error vector is exactly zero),
        ``update_norm``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 9, Hebb's Rule section.

    Examples
    --------
    One input, one output, predicted 0 where 1 was wanted:

    >>> r = geron_hebb_rule([1.0, 2.0], [1.0], [0.0], [[0.0], [0.0]], eta=0.1)
    >>> r["w_new"]
    [[0.1], [0.2]]
    >>> r["converged"]
    False

    Get it right and the weights are frozen -- the defining property of
    the rule:

    >>> r2 = geron_hebb_rule([1.0, 2.0], [1.0], [1.0], [[0.1], [0.2]], eta=0.1)
    >>> r2["w_new"]
    [[0.1], [0.2]]
    >>> r2["converged"]
    True
    """
    x = np.asarray(x, dtype=float).ravel()
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    W = np.atleast_2d(np.asarray(w, dtype=float))
    if y_true.size != y_pred.size:
        raise ValueError(f"y_true has {y_true.size} entries but y_pred has {y_pred.size}.")
    if W.shape != (x.size, y_true.size):
        raise ValueError(
            f"w must have shape (n_in, n_out) = ({x.size}, {y_true.size}), got {W.shape}."
        )
    eta = float(eta)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"eta must be a positive finite float, got {eta}.")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y_true))
            and np.all(np.isfinite(y_pred)) and np.all(np.isfinite(W))):
        raise ValueError("x, y_true, y_pred and w must all be finite.")

    err = y_true - y_pred
    dW = eta * np.outer(x, err)
    W_new = W + dW

    return RichResult(
        title="Hebb / perceptron update",
        summary_lines=[("||error||", float(np.linalg.norm(err))),
                       ("||dW||", float(np.linalg.norm(dW)))],
        payload={
            "w_new": W_new.tolist(),
            "delta_w": dW.tolist(),
            "error": err.tolist(),
            "converged": bool(np.all(err == 0.0)),
            "update_norm": float(np.linalg.norm(dW)),
            "eta": eta,
            "estimate": W_new.tolist(),
            "n": int(x.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grhbb: w += eta * outer(x, y_true - y_pred); no error, no update"
