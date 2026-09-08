# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""HMM forward algorithm."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wasserman_hmm_forward"]


def wasserman_hmm_forward(obs, A, B, pi):
    """
    HMM forward algorithm with per-step scaling.

    Formula: alpha_t(j) = [sum_i alpha_{t-1}(i) a_ij] b_j(o_t),
    alpha_1(j) = pi_j b_j(o_1). Each step is normalised and the log
    of the scale factors accumulates the exact log-likelihood
    log P(o_1..o_T), so long sequences do not underflow. Rows of A,
    B and pi must each sum to 1 within 1e-8.

    Parameters
    ----------
    obs : array-like of int
        Observation indices (0-based), length T >= 1.
    A : array-like (S, S)
        Transition matrix, rows sum to 1.
    B : array-like (S, M)
        Emission matrix, rows sum to 1.
    pi : array-like (S,)
        Initial distribution.

    Returns
    -------
    result : dict
        Keys: estimate (log-likelihood), filtered (posterior of the
        final state, per state), T, S, method.

    References
    ----------
    Wasserman (2004), Ch 23 (Probability Redux: Stochastic Processes --
    Markov chains); the HMM forward algorithm follows Rabiner (1989).

    Examples
    --------
    A deterministic chain emits its symbols with probability 1:

    >>> A = [[0.0, 1.0], [1.0, 0.0]]
    >>> B = [[1.0, 0.0], [0.0, 1.0]]
    >>> out = wasserman_hmm_forward([0, 1, 0], A, B, [1.0, 0.0])
    >>> round(out["estimate"], 12)
    0.0
    >>> out["filtered"]
    [1.0, 0.0]
    >>> import math
    >>> iid = wasserman_hmm_forward([0, 0], [[1.0]], [[0.5, 0.5]], [1.0])
    >>> abs(iid["estimate"] - 2 * math.log(0.5)) < 1e-15
    True
    >>> wasserman_hmm_forward([2], A, B, [1.0, 0.0])
    Traceback (most recent call last):
        ...
    ValueError: observation index 2 is outside the emission alphabet of size 2.
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
    for name, mat in (("A", A), ("B", B)):
        if np.any(np.abs(np.sum(mat, axis=1) - 1.0) > 1e-8):
            raise ValueError(f"rows of {name} must sum to 1.")
    if abs(float(np.sum(pi)) - 1.0) > 1e-8:
        raise ValueError("pi must sum to 1.")
    if not obs:
        raise ValueError("the forward algorithm needs at least one observation.")
    for o in obs:
        if not 0 <= o < M:
            raise ValueError(f"observation index {o} is outside the emission alphabet of size {M}.")
    alpha = pi * B[:, obs[0]]
    ll = 0.0
    c = float(np.sum(alpha))
    if c == 0:
        return RichResult(payload={"estimate": float("-inf"),
                                   "filtered": [0.0] * S, "T": len(obs),
                                   "S": int(S), "method": "forward (impossible sequence)"})
    alpha /= c
    ll += np.log(c)
    for o in obs[1:]:
        alpha = (alpha @ A) * B[:, o]
        c = float(np.sum(alpha))
        if c == 0:
            return RichResult(payload={"estimate": float("-inf"),
                                       "filtered": [0.0] * S, "T": len(obs),
                                       "S": int(S), "method": "forward (impossible sequence)"})
        alpha /= c
        ll += np.log(c)
    return RichResult(payload={
        "estimate": float(ll), "filtered": [float(v) for v in alpha],
        "T": len(obs), "S": int(S),
        "method": "scaled forward algorithm; exact log-likelihood"})


def cheatsheet():
    return "wsmhmm: scaled alpha recursion; ll = sum log scale factors"
