# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mamba selective state-space recurrence (S6)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kamath_mamba_ssm"]


def _per_step(v, T, N, name):
    """Accept (N,) shared across time or (T, N) input-dependent."""
    a = np.asarray(v, dtype=float)
    if a.ndim == 1:
        if a.size != N:
            raise ValueError(f"{name} must have {N} entries; got {a.size}.")
        return np.repeat(a[None, :], T, axis=0)
    if a.shape != (T, N):
        raise ValueError(f"{name} must be ({T}, {N}) or ({N},); got {a.shape}.")
    return a


def kamath_mamba_ssm(x, A, B, C, delta):
    """h_t = Abar(x_t) h_{t-1} + Bbar(x_t) x_t;  y_t = C(x_t) h_t.

    Selective means B, C and delta may vary with the token: pass them
    as (T, N) for the input-dependent case or (N,) to share them.
    Discretisation is zero-order hold on a DIAGONAL A, which is what
    makes the recurrence a scan instead of a matrix exponential:
    Abar_t = exp(delta_t * A) elementwise and Bbar_t = delta_t * B_t.
    A full (N, N) A is refused rather than silently diagonalised.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 10,
    Mamba / selective SSM; the 2024 PDF mentions Mamba but carries no
    equation for it, so the recurrence is implemented exactly as the
    spec line states (Gu and Dao 2023).

    Examples
    --------
    >>> out = kamath_mamba_ssm([1.0, 2.0, 3.0], [0.0], [1.0], [1.0], 1.0)
    >>> out["y"]
    [1.0, 3.0, 6.0]
    >>> out2 = kamath_mamba_ssm([1.0, 1.0], [-1.0], [1.0], [1.0], 1.0)
    >>> import math
    >>> abs(out2["y"][1] - (math.exp(-1.0) + 1.0)) < 1e-12
    True
    """
    x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    A = np.asarray(A, dtype=float)
    if A.ndim == 2:
        raise ValueError(
            "A must be the diagonal of the state matrix, shape (N,); a "
            "dense (N, N) A needs a matrix exponential, which is not "
            "the selective-scan parameterisation.")
    A = np.atleast_1d(A).ravel()
    T, N = x.size, A.size
    if T == 0:
        raise ValueError("the input sequence is empty.")
    if N == 0:
        raise ValueError("the state dimension is 0.")
    Bm = _per_step(B, T, N, "B")
    Cm = _per_step(C, T, N, "C")
    d = np.asarray(delta, dtype=float)
    if d.ndim == 0:
        d = np.full(T, float(d))
    else:
        d = d.ravel()
        if d.size != T:
            raise ValueError(
                f"delta must be scalar or have {T} entries; got {d.size}.")
    if np.any(d <= 0):
        raise ValueError(
            "delta is a step size and must be positive; a non-positive "
            "step inverts or freezes the discretisation.")

    Abar = np.exp(d[:, None] * A[None, :])
    Bbar = d[:, None] * Bm
    h = np.zeros(N)
    ys, hs = [], []
    for t in range(T):
        h = Abar[t] * h + Bbar[t] * x[t]
        hs.append(h.copy())
        ys.append(float(np.dot(Cm[t], h)))
    return RichResult(payload={
        "y": ys,
        "states": [[float(v) for v in row] for row in hs],
        "A_bar": [[float(v) for v in row] for row in Abar],
        "estimate": ys[-1], "state_dim": N, "n": T,
        "method": "Mamba selective SSM scan (ZOH, diagonal A)"})


def cheatsheet():
    return "kmmamb: h_t = exp(dt*A)h_{t-1} + dt*B x_t; y_t = C.h_t"
