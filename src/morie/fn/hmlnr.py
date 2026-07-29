# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer normalization applied in RNN cells."""

import numpy as np

from ._richresult import RichResult
from .hmlntr import geron_layer_normalization

__all__ = ["geron_layer_norm_rnn"]

_METHOD = "Layer normalization inside an RNN cell"


def geron_layer_norm_rnn(x, gamma=1.0, beta=0.0, eps=1e-5, activation="tanh"):
    """
    Layer normalization applied in RNN cells.

    Formula: normalize across feature dim within each time step

    The normalization itself is delegated to
    :func:`morie.fn.hmlntr.geron_layer_normalization`; what this entry
    adds is the RNN placement.  Inside a cell the norm goes **after the
    linear combination and before the activation**, once per time step,
    using only that step's own features.

    That ordering matters: batch normalization cannot be used here
    because its statistics would have to be accumulated separately for
    every time step (and a sequence longer than any seen in training
    would have none), whereas layer norm depends on nothing but the
    current step. Applying it after the activation instead would undo
    the saturation control it exists to provide.

    ``x`` is one pre-activation per time step: rows are time steps,
    columns are units.

    Parameters
    ----------
    x : array-like, shape (T, n_units) or (n_units,)
        Pre-activations ``W_x x_t + W_h h_{t-1} + b`` for each step.
    gamma, beta : float or array-like of length n_units
        Per-unit scale and shift, shared across time steps.
    eps : float
        Variance floor.
    activation : {"tanh", "relu", "none"}
        Applied after the normalization.

    Returns
    -------
    result : RichResult
        Keys: h, normalized, mu, var, estimate, n, method.

    Examples
    --------
    Two time steps at wildly different scales normalize to the same
    values -- that scale invariance per step is the point:

    >>> r = geron_layer_norm_rnn([[1.0, 3.0], [100.0, 300.0]], eps=0.0, activation="none")
    >>> [[round(float(v), 9) for v in row] for row in r["normalized"]]
    [[-1.0, 1.0], [-1.0, 1.0]]

    With tanh the output is ``tanh(-1), tanh(1)``:

    >>> t = geron_layer_norm_rnn([[1.0, 3.0]], eps=0.0)
    >>> [round(float(v), 9) for v in t["h"].ravel()]
    [-0.761594156, 0.761594156]

    The statistics are per time step, so the second row's mean is 200:

    >>> [float(v) for v in r["mu"]]
    [2.0, 200.0]

    References
    ----------
    Géron Ch 13
    """
    if activation not in ("tanh", "relu", "none"):
        raise ValueError(f"geron_layer_norm_rnn: activation must be 'tanh', 'relu' or 'none', got {activation!r}")
    inner = geron_layer_normalization(x, gamma=gamma, beta=beta, eps=eps)
    z = inner["y"]
    if activation == "tanh":
        h = np.tanh(z)
    elif activation == "relu":
        h = np.maximum(z, 0.0)
    else:
        h = z

    return RichResult(
        title="Layer-normalized RNN step",
        summary_lines=[
            ("Time steps", int(z.shape[0])),
            ("Units", int(z.shape[1])),
            ("Activation", activation),
        ],
        interpretation=(
            "Normalize after the linear combination and before the activation; batch norm cannot "
            "be used here because its statistics are per time step and do not generalise to unseen lengths."
        ),
        payload={
            "h": h,
            "normalized": inner["x_hat"],
            "pre_activation": z,
            "mu": inner["mu"],
            "var": inner["var"],
            "estimate": float(np.mean(h)),
            "n": int(z.shape[0]),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "hmlnr: layer norm inside an RNN cell -- per time step, before the activation (delegates to hmlntr)"
