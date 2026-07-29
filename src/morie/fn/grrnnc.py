# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Simple RNN cell forward step."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_simple_rnn_cell"]

_METHOD = "Simple (Elman) RNN cell"


def geron_simple_rnn_cell(x_t, h_prev, Whh, Wxh, b):
    r"""One recurrent step.

    .. math::
        \mathbf{h}_t = \tanh\bigl(W_{hh}\mathbf{h}_{t-1}
                       + W_{xh}\mathbf{x}_t + \mathbf{b}\bigr)

    The same :math:`W_{hh}` is applied at every step, so the gradient
    through :math:`T` steps carries :math:`W_{hh}^{T}`: the spectral
    radius of that one matrix decides between vanishing and exploding,
    which is the entire motivation for LSTM/GRU.  The largest singular
    value is reported for that reason.  tanh saturation is reported too
    -- once a unit is pinned at :math:`\pm 1` its local derivative
    :math:`1 - h^2` is nearly zero and it stops learning.

    Parameters
    ----------
    x_t : array-like, shape (n_in,)
    h_prev : array-like, shape (n_h,)
    Whh : array-like, shape (n_h, n_h)
    Wxh : array-like, shape (n_h, n_in)
    b : array-like, shape (n_h,)

    Returns
    -------
    RichResult
        Payload keys ``h``, ``pre_activation``, ``derivative``,
        ``spectral_norm_Whh``, ``saturated``, ``estimate``, ``n``,
        ``method``.

    References
    ----------
    Géron Ch 13, Eq 13-1 (Simple RNN cell).

    Examples
    --------
    Identity recurrence, zero input weights, ``h_prev = [0.5, -0.5]``:
    the cell just squashes the previous state.

    >>> I = [[1.0, 0.0], [0.0, 1.0]]
    >>> Z = [[0.0], [0.0]]
    >>> r = geron_simple_rnn_cell([1.0], [0.5, -0.5], I, Z, [0.0, 0.0])
    >>> [round(v, 6) for v in r["h"]]
    [0.462117, -0.462117]

    Input weights do reach the state:

    >>> r2 = geron_simple_rnn_cell([1.0], [0.0, 0.0], I, [[2.0], [0.0]], [0.0, 0.0])
    >>> [round(v, 6) for v in r2["h"]]
    [0.964028, 0.0]
    """
    x = np.asarray(x_t, dtype=float).ravel()
    h = np.asarray(h_prev, dtype=float).ravel()
    A = np.atleast_2d(np.asarray(Whh, dtype=float))
    B = np.atleast_2d(np.asarray(Wxh, dtype=float))
    bv = np.asarray(b, dtype=float).ravel()
    n_h = h.size
    if n_h == 0 or x.size == 0:
        raise ValueError("x_t and h_prev must both be non-empty.")
    if A.shape != (n_h, n_h):
        raise ValueError(f"Whh must be ({n_h}, {n_h}) to match h_prev, got {A.shape}.")
    if B.shape != (n_h, x.size):
        raise ValueError(f"Wxh must be ({n_h}, {x.size}), got {B.shape}.")
    if bv.size != n_h:
        raise ValueError(f"b must have {n_h} entries, got {bv.size}.")
    for name, M in (("x_t", x), ("h_prev", h), ("Whh", A), ("Wxh", B), ("b", bv)):
        if not np.all(np.isfinite(M)):
            raise ValueError(f"{name} contains non-finite values.")

    z = A @ h + B @ x + bv
    h_new = np.tanh(z)
    sv = float(np.linalg.svd(A, compute_uv=False).max())

    return RichResult(
        title="Simple RNN cell",
        summary_lines=[("Hidden units", n_h), ("||Whh||_2", sv)],
        payload={
            "h": h_new.tolist(),
            "pre_activation": z.tolist(),
            "derivative": (1.0 - h_new**2).tolist(),
            "spectral_norm_Whh": sv,
            "saturated": float(np.mean(np.abs(h_new) > 0.99)),
            "estimate": h_new.tolist(),
            "n": int(n_h),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grrnnc: h_t = tanh(Whh h_{t-1} + Wxh x_t + b); ||Whh||_2 decides vanish vs explode"
