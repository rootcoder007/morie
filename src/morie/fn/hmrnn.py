# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recurrent neuron step."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_recurrent_neuron"]

_ACT = {
    "tanh": (np.tanh, lambda a: 1.0 - a * a),
    "relu": (lambda z: np.maximum(z, 0.0), lambda a: (a > 0).astype(float)),
    "sigmoid": (lambda z: 1.0 / (1.0 + np.exp(-z)), lambda a: a * (1.0 - a)),
    "identity": (lambda z: z, lambda a: np.ones_like(a)),
}


def geron_recurrent_neuron(x_t, h_prev, Wx, Wh, b, activation="tanh"):
    """
    Recurrent neuron step: hidden state updated from the previous state.

    Formula: h_t = phi(W_x x_t + W_h h_{t-1} + b)

    The same weights are reused at every timestep, which is what lets one
    cell read a sequence of any length -- and what makes the gradient a
    PRODUCT of Jacobians down the unrolled chain. The spectral norm of
    dh_t/dh_{t-1} is returned for that reason: repeatedly below 1 and
    gradients vanish, repeatedly above 1 and they explode.

    Parameters
    ----------
    x_t : array-like, shape (n_in,)
        Input at this step.
    h_prev : array-like, shape (n_units,)
        Previous hidden state.
    Wx : array-like, shape (n_units, n_in)
    Wh : array-like, shape (n_units, n_units)
    b : array-like, shape (n_units,)
    activation : {"tanh", "relu", "sigmoid", "identity"}, default "tanh"

    Returns
    -------
    result : RichResult
        Keys: h, z, jacobian_norm, estimate, n, method.

    Examples
    --------
    >>> r = geron_recurrent_neuron([1.0], [0.0], [[1.0]], [[1.0]], [0.0])
    >>> round(float(r["h"][0]), 6)
    0.761594

    Feeding that state back with a zero input gives tanh(0.761594...):

    >>> r2 = geron_recurrent_neuron([0.0], r["h"], [[1.0]], [[1.0]], [0.0])
    >>> round(float(r2["h"][0]), 6)
    0.642015

    References
    ----------
    Geron Ch 13
    """
    if activation not in _ACT:
        raise ValueError(f"geron_recurrent_neuron: activation must be one of {sorted(_ACT)}, got {activation!r}")
    x = np.atleast_1d(np.asarray(x_t, dtype=float)).ravel()
    h = np.atleast_1d(np.asarray(h_prev, dtype=float)).ravel()
    A = np.atleast_2d(np.asarray(Wx, dtype=float))
    B = np.atleast_2d(np.asarray(Wh, dtype=float))
    bb = np.atleast_1d(np.asarray(b, dtype=float)).ravel()
    u = h.size
    if A.shape != (u, x.size):
        raise ValueError(f"geron_recurrent_neuron: Wx has shape {A.shape}, expected {(u, x.size)}")
    if B.shape != (u, u):
        raise ValueError(f"geron_recurrent_neuron: Wh has shape {B.shape}, expected {(u, u)}")
    if bb.size != u:
        raise ValueError(f"geron_recurrent_neuron: b has {bb.size} entries but there are {u} units")
    for name, arr in (("x_t", x), ("h_prev", h), ("Wx", A), ("Wh", B), ("b", bb)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"geron_recurrent_neuron: {name} contains non-finite values")

    phi, dphi = _ACT[activation]
    z = A @ x + B @ h + bb
    hn = phi(z)
    jac = (dphi(hn)[:, None]) * B
    return RichResult(
        title="Recurrent neuron step",
        summary_lines=[("Units", int(u)), ("Jacobian spectral norm", float(np.linalg.norm(jac, 2)))],
        interpretation="Weights are shared across time, so the gradient is a product of these Jacobians.",
        payload={
            "h": hn,
            "h_next": hn,
            "z": z,
            "jacobian": jac,
            "jacobian_norm": float(np.linalg.norm(jac, 2)),
            "estimate": hn,
            "n": int(u),
            "method": f"Recurrent step h_t = {activation}(Wx x + Wh h + b)",
        },
    )


def cheatsheet():
    return "hmrnn: Recurrent neuron step h_t = phi(Wx x_t + Wh h_{t-1} + b)"
