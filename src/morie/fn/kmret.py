# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RetNet retention: the parallel form of a decaying recurrence."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kamath_retnet_retention"]


def kamath_retnet_retention(Q, K, V, gamma):
    """Retention(Q, K, V) = ((Q K^T) .* D) V, with
    D_ij = gamma^(i-j) for i >= j and 0 otherwise.

    No softmax and no 1/sqrt(d) -- that is the point: dropping the
    normaliser is what lets the same computation be written as an O(1)
    recurrent step at inference. D is causal by construction, so the
    lower triangle carries an exponentially decaying weight on
    distance and the upper triangle is exactly 0.

    The recurrent form ``S_t = gamma * S_{t-1} + K_t^T V_t``,
    ``y_t = Q_t S_t``, is returned alongside as ``states`` so the two
    forms can be checked against each other rather than assumed equal.

    Reference: the worklist cites Kamath, Keenan, Somers and Sorenson
    (2024), *Large Language Models: A Deep Dive*, Springer, Ch 10,
    RetNet; that section is not in the 2024 PDF, so retention is
    implemented exactly as the spec line states (Sun et al. 2023).

    Examples
    --------
    >>> out = kamath_retnet_retention([[1.0], [1.0]], [[1.0], [1.0]],
    ...                               [[1.0], [2.0]], 0.5)
    >>> out["output"]
    [[1.0], [2.5]]
    >>> out["decay"]
    [[1.0, 0.0], [0.5, 1.0]]
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    K = np.atleast_2d(np.asarray(K, dtype=float))
    V = np.atleast_2d(np.asarray(V, dtype=float))
    gamma = float(gamma)
    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            f"Q and K must share a width; got {Q.shape[1]} and "
            f"{K.shape[1]}.")
    if Q.shape[0] != K.shape[0] or K.shape[0] != V.shape[0]:
        raise ValueError(
            "retention is causal within one sequence: Q, K and V must "
            f"have the same length; got {Q.shape[0]}, {K.shape[0]}, "
            f"{V.shape[0]}.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(
            f"gamma must lie in (0, 1]; got {gamma}. Outside that range "
            "the decay grows with distance instead of shrinking.")
    T = Q.shape[0]
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    D = np.where(i >= j, gamma ** (i - j), 0.0)
    out = (Q @ K.T * D) @ V

    # Recurrent form, for cross-checking.
    S = np.zeros((Q.shape[1], V.shape[1]))
    rec = []
    for t in range(T):
        S = gamma * S + np.outer(K[t], V[t])
        rec.append(Q[t] @ S)
    return RichResult(payload={
        "output": [[float(v) for v in row] for row in out],
        "decay": [[float(v) for v in row] for row in D],
        "recurrent_output": [[float(v) for v in row] for row in rec],
        "estimate": float(out[-1, 0]),
        "gamma": gamma, "n": T,
        "method": "RetNet retention ((QK^T) .* D) V"})


def cheatsheet():
    return "kmret: (QK^T .* gamma^(i-j) causal) V; no softmax, no 1/sqrt(d)"
