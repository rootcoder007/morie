# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Backpropagation through time: unroll the RNN and apply standard backprop."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_backprop_through_time"]

_METHOD = "Backpropagation through time (tanh RNN)"


def geron_backprop_through_time(loss_grads, hiddens, inputs, W_h=None, h_init=None):
    r"""Accumulate RNN weight gradients across an unrolled sequence.

    With the row-vector convention
    :math:`h_t = \tanh(x_t W_x + h_{t-1} W_h)`, so that ``W_x`` is
    ``(D, H)`` and ``W_h`` is ``(H, H)``,

    .. math::
        \delta_t = \Bigl(\frac{\partial L_t}{\partial h_t}
                 + W_h \delta_{t+1}\Bigr)\odot (1 - h_t^2),
        \qquad
        \nabla_{W_x} L = \sum_t x_t \delta_t^{\top},\quad
        \nabla_{W_h} L = \sum_t h_{t-1}\delta_t^{\top}

    The single shared weight matrix picks up a contribution at *every*
    time step -- that summation is the whole difference from ordinary
    backprop.  It is also where the vanishing gradient comes from: the
    recurrent term multiplies by :math:`W_h` and by
    :math:`1 - h_t^2 \le 1` once per step, so the reported
    ``per_step_delta_norm`` typically decays geometrically backwards.

    Parameters
    ----------
    loss_grads : array-like, shape (T, H)
        :math:`\partial L_t / \partial h_t`, the *direct* loss gradient
        at each step (excluding the recurrent path).
    hiddens : array-like, shape (T, H)
        Hidden states :math:`h_1 \dots h_T`.
    inputs : array-like, shape (T, D)
        Inputs :math:`x_1 \dots x_T`.
    W_h : array-like, shape (H, H), optional
        Recurrent weights, in the ``h_{t-1} @ W_h`` convention above.
        If omitted the recurrent path is dropped and
        ``delta_t`` uses ``loss_grads[t]`` alone -- useful for checking
        the per-step terms in isolation.
    h_init : array-like, shape (H,), optional
        State before the first step, default zeros.

    Returns
    -------
    RichResult
        Payload keys ``grad_Wx``, ``grad_Wh``, ``grad_b``, ``deltas``,
        ``per_step_delta_norm``, ``vanishing_ratio``, ``estimate``
        (Frobenius norm of ``grad_Wh``), ``n``, ``method``.

    References
    ----------
    Géron Ch 13, Backpropagation Through Time section.

    Examples
    --------
    Zero hidden states make ``1 - h^2 = 1``, so the deltas are the loss
    gradients and ``grad_Wx`` is just ``sum_t x_t``:

    >>> r = geron_backprop_through_time([[1.0], [1.0]], [[0.0], [0.0]],
    ...                                 [[1.0], [2.0]], h_init=[1.0])
    >>> r["grad_Wx"]
    [[3.0]]
    >>> r["grad_Wh"]
    [[1.0]]

    Supplying ``W_h`` adds the recurrent path, so the earlier delta grows:

    >>> r2 = geron_backprop_through_time([[1.0], [1.0]], [[0.0], [0.0]],
    ...                                  [[1.0], [2.0]], W_h=[[0.5]])
    >>> r2["deltas"]
    [[1.5], [1.0]]
    >>> r2["per_step_delta_norm"]
    [1.5, 1.0]
    """
    G = np.atleast_2d(np.asarray(loss_grads, dtype=float))
    H = np.atleast_2d(np.asarray(hiddens, dtype=float))
    X = np.atleast_2d(np.asarray(inputs, dtype=float))
    if G.shape != H.shape:
        raise ValueError(
            f"loss_grads shape {G.shape} must match hiddens shape {H.shape}."
        )
    if X.shape[0] != H.shape[0]:
        raise ValueError(
            f"inputs has {X.shape[0]} time steps but hiddens has {H.shape[0]}."
        )
    if H.size == 0:
        raise ValueError("no time steps supplied.")
    for name, arr in (("loss_grads", G), ("hiddens", H), ("inputs", X)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values.")
    T, hdim = H.shape
    if np.any(np.abs(H) > 1.0 + 1e-9):
        raise ValueError(
            "hiddens fall outside [-1, 1]; this routine assumes a tanh RNN, so "
            "1 - h^2 would be negative."
        )
    if W_h is None:
        Wh = None
    else:
        Wh = np.atleast_2d(np.asarray(W_h, dtype=float))
        if Wh.shape != (hdim, hdim):
            raise ValueError(f"W_h must have shape {(hdim, hdim)}, got {Wh.shape}.")
        if not np.all(np.isfinite(Wh)):
            raise ValueError("W_h contains non-finite values.")
    h_prev0 = np.zeros(hdim) if h_init is None else np.asarray(h_init, dtype=float).ravel()
    if h_prev0.size != hdim:
        raise ValueError(f"h_init must have {hdim} entries, got {h_prev0.size}.")

    deltas = np.zeros((T, hdim))
    carry = np.zeros(hdim)
    for t in range(T - 1, -1, -1):
        upstream = G[t] + carry
        deltas[t] = upstream * (1.0 - H[t] ** 2)
        carry = deltas[t] @ Wh.T if Wh is not None else np.zeros(hdim)

    H_prev = np.vstack([h_prev0[None, :], H[:-1]])
    grad_Wx = X.T @ deltas
    grad_Wh = H_prev.T @ deltas
    grad_b = deltas.sum(axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    ratio = float(norms[0] / norms[-1]) if norms[-1] > 0 else float("nan")

    return RichResult(
        title="Backprop through time",
        summary_lines=[("Time steps", T), ("‖grad_Wh‖_F", float(np.linalg.norm(grad_Wh)))],
        payload={
            "grad_Wx": grad_Wx.tolist(),
            "grad_Wh": grad_Wh.tolist(),
            "grad_b": grad_b.tolist(),
            "deltas": deltas.tolist(),
            "per_step_delta_norm": norms.tolist(),
            "vanishing_ratio": ratio,
            "estimate": float(np.linalg.norm(grad_Wh)),
            "n": int(T),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbptt: BPTT -- delta_t=(dL/dh_t + W_h delta_{t+1})*(1-h_t^2); grads summed over time"
