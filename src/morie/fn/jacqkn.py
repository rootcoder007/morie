# morie.fn -- function file (rootcoder007/morie)
"""Jacquez k-nearest-neighbour test for space-time interaction."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["jacquez_k_nn_test"]


def _knn_indicator(D, k):
    """a_ij = 1 when j is among i's k nearest, excluding i itself."""
    n = D.shape[0]
    M = D.copy()
    np.fill_diagonal(M, np.inf)
    # argsort takes the k smallest per row; ties break by index, which is
    # the same convention Jacquez's own implementation uses.
    nn = np.argsort(M, axis=1, kind="stable")[:, :k]
    A = np.zeros((n, n), dtype=bool)
    np.put_along_axis(A, nn, True, axis=1)
    return A


def jacquez_k_nn_test(coords, time, k=3, cdf=None, B=999, seed=None):
    r"""Jacquez's k-nearest-neighbour test for space-time interaction.

    Counts the pairs that are near in space *and* near in time:

    .. math::

        J_k = \sum_i \sum_j a_{ij}^{(k)} b_{ij}^{(k)}

    where :math:`a_{ij}^{(k)}` is 1 when case :math:`j` is among case
    :math:`i`'s :math:`k` nearest neighbours in space, and
    :math:`b_{ij}^{(k)}` is 1 when it is among the :math:`k` nearest in
    time. Neither indicator is symmetric -- being your neighbour does not
    make you mine -- so the double sum runs over ordered pairs.

    The statistic is a count, not a distance, so it does not depend on
    the units of either the coordinates or the clock, and it stays valid
    for irregular study regions and populations at risk that vary in
    space. That distribution-freeness is the reason to use it, and it is
    also why the null must be simulated: under the null of no
    interaction the time labels are exchangeable, so permuting them and
    ranking the observed count gives

    .. math:: p = \frac{1 + \#\{J_k^{(b)} \ge J_k^{obs}\}}{1 + B}

    Parameters
    ----------
    coords : array-like, shape (n, d)
        Case locations.
    time : array-like, shape (n,)
        Case times, on any scale where nearness means what you intend.
    k : int, default 3
        Number of nearest neighbours, in space and in time alike. Must
        be at least 1 and less than n.
    cdf : callable, optional
        Null CDF of the statistic, replacing the permutation null.
    B : int, default 999
        Number of time-label permutations.
    seed : int, optional
        Seed for the permutations.

    Returns
    -------
    RichResult
        keys: ``statistic`` (J_k), ``p_value``, ``expected`` (mean under
        the permutation null), ``k``, ``n``, ``B``, ``null_statistics``,
        ``method``.

    References
    ----------
    Jacquez, G. M. (1996). A k nearest neighbour test for space-time
    interaction. *Statistics in Medicine*, 15(18), 1935-1949.
    """
    P = np.asarray(coords, dtype=float)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if P.ndim != 2:
        raise ValueError(f"coords must be (n, d); got shape {P.shape}.")
    t = np.asarray(time, dtype=float).ravel()
    n = P.shape[0]
    if t.size != n:
        raise ValueError(f"time must have one entry per case; got {t.size} and {n}.")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(t))):
        raise ValueError("coords and time must be finite.")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")
    if k >= n:
        raise ValueError(f"k must be smaller than the number of cases; got k={k}, n={n}.")

    Ds = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=-1))
    A = _knn_indicator(Ds, k)

    def stat(times):
        Dt = np.abs(times[:, None] - times[None, :])
        return int(np.sum(A & _knn_indicator(Dt, k)))

    observed = stat(t)

    if cdf is not None:
        return RichResult(
            title="Jacquez k-nearest-neighbour space-time test",
            payload={
                "statistic": observed,
                "p_value": float(1.0 - cdf(observed)),
                "expected": None,
                "k": k,
                "n": int(n),
                "B": 0,
                "method": "Jacquez k-NN test against a supplied null CDF",
            },
        )

    B = int(B)
    if B < 1:
        raise ValueError(f"B must be at least 1, got {B}.")
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    for b in range(B):
        null[b] = stat(rng.permutation(t))

    p = (1.0 + float(np.sum(null >= observed))) / (1.0 + B)

    return RichResult(
        title="Jacquez k-nearest-neighbour space-time test",
        payload={
            "statistic": observed,
            "p_value": p,
            "expected": float(null.mean()),
            "k": k,
            "n": int(n),
            "B": B,
            "null_statistics": null,
            "method": "Jacquez (1996) k-NN space-time interaction test",
        },
    )


def cheatsheet():
    return "jacqkn: Jacquez k-nearest-neighbour space-time interaction test"
