# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Viterbi decoding."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_viterbi"]


def wasserman_viterbi(obs, A, B, pi):
    """
    Viterbi most-probable state path.

    Formula: delta_t(j) = max_i delta_{t-1}(i) a_ij b_j(o_t), with
    backpointers; computed in log space so long sequences cannot
    underflow, ties broken toward the LOWER state index
    (deterministic, documented). Validation matches wsmhmm.

    Parameters
    ----------
    obs : array-like of int
        Observation indices (0-based).
    A, B, pi
        As in wsmhmm.

    Returns
    -------
    result : dict
        Keys: estimate (log probability of the best path), path
        (0-based states), T, S, method.

    References
    ----------
    Wasserman (2004), Ch 23; Viterbi (1967).

    Examples
    --------
    >>> A = [[0.0, 1.0], [1.0, 0.0]]
    >>> B = [[1.0, 0.0], [0.0, 1.0]]
    >>> out = wasserman_viterbi([0, 1, 0], A, B, [1.0, 0.0])
    >>> out["path"]
    [0, 1, 0]
    >>> round(out["estimate"], 12)
    0.0
    >>> noisy = wasserman_viterbi([0, 0], [[0.9, 0.1], [0.1, 0.9]],
    ...                           [[0.8, 0.2], [0.2, 0.8]], [0.5, 0.5])
    >>> noisy["path"]
    [0, 0]
    """
    obs = [int(v) for v in np.atleast_1d(np.asarray(obs))]
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    pi = np.atleast_1d(np.asarray(pi, dtype=float))
    S, M = B.shape
    if A.shape != (S, S):
        raise ValueError(f"A must be {S}x{S} to match B's rows; got {A.shape}.")
    if pi.size != S:
        raise ValueError(f"pi must have {S} entries; got {pi.size}.")
    if not obs:
        raise ValueError("Viterbi needs at least one observation.")
    for o in obs:
        if not 0 <= o < M:
            raise ValueError(f"observation index {o} is outside the emission alphabet of size {M}.")
    with np.errstate(divide="ignore"):
        lA, lB, lpi = np.log(A), np.log(B), np.log(pi)
    T = len(obs)
    delta = lpi + lB[:, obs[0]]
    back = np.zeros((T, S), dtype=int)
    for t in range(1, T):
        cand = delta[:, None] + lA
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(S)] + lB[:, obs[t]]
    end = int(np.argmax(delta))
    path = [end]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t][path[-1]]))
    path.reverse()
    return RichResult(payload={
        "estimate": float(delta[end]), "path": path, "T": T, "S": int(S),
        "method": "log-space Viterbi, ties to lower state index"})


def cheatsheet():
    return "wsmvit: log delta recursion + backpointers; argmax ties -> lower index"
