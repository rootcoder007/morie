# morie.fn -- function file (rootcoder007/morie)
"""Permutation two-sample test on the 1-Wasserstein distance."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ot_permutation_test_w1"]


def _w1(a, b):
    r"""1-Wasserstein distance between two empirical measures on the line.

    On :math:`\mathbb{R}` the optimal transport cost for :math:`p = 1` has
    the closed form

    .. math:: W_1(F, G) = \int_{-\infty}^{\infty} |F(x) - G(x)|\,dx

    so no linear program is needed. The integral is evaluated on the
    merged support: between consecutive distinct values the two step
    functions are constant, so each strip contributes
    ``|F - G| * width``.
    """
    xs = np.sort(a)
    ys = np.sort(b)
    grid = np.concatenate([xs, ys])
    grid.sort()
    width = np.diff(grid)
    if width.size == 0:
        return 0.0
    F = np.searchsorted(xs, grid[:-1], side="right") / xs.size
    G = np.searchsorted(ys, grid[:-1], side="right") / ys.size
    return float(np.sum(np.abs(F - G) * width))


def ot_permutation_test_w1(X, Y, B=999, cdf=None, seed=None):
    r"""Two-sample permutation test on the 1-Wasserstein distance.

    Uses :math:`W_1` as the statistic, which on the line is

    .. math:: W_1(F, G) = \int |F(x) - G(x)|\,dx

    the area between the two empirical distribution functions. That
    closed form is why this test needs no optimal-transport solver: the
    one-dimensional transport problem is solved by sorting.

    :math:`W_1` has no tractable null distribution, so the p-value comes
    from permuting the group labels: pool the two samples, re-split at
    the original sizes ``B`` times, and rank the observed distance among
    the permuted ones,

    .. math:: p = \frac{1 + \#\{W_1^{(b)} \ge W_1^{obs}\}}{1 + B}

    The test is sensitive to any difference in distribution, not only in
    location, because :math:`W_1` integrates the whole gap between the
    two CDFs rather than comparing summaries.

    Parameters
    ----------
    X, Y : array-like
        The two samples, one-dimensional. Multi-dimensional input is
        rejected rather than silently flattened: the closed form above
        holds only on the line, and the general case needs a transport
        solver this function does not carry.
    B : int, default 999
        Number of label permutations.
    cdf : callable, optional
        Null CDF of the statistic, replacing the permutation null.
    seed : int, optional
        Seed for the permutations.

    Returns
    -------
    RichResult
        keys: ``statistic`` (W_1), ``p_value``, ``B``, ``m``, ``n``,
        ``null_statistics``, ``method``.

    References
    ----------
    Villani, C. (2009). *Optimal Transport: Old and New*. Grundlehren der
    mathematischen Wissenschaften 338. Springer, Berlin. Theorem 2.18
    gives the one-dimensional form.

    Ramdas, A., Garcia Trillos, N. & Cuturi, M. (2017). On Wasserstein
    two-sample testing and related families of nonparametric tests.
    *Entropy*, 19(2), 47.
    """
    a = np.asarray(X, dtype=float)
    b = np.asarray(Y, dtype=float)
    for arr, name in ((a, "X"), (b, "Y")):
        if arr.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional; the closed-form W_1 used here holds only on the line, "
                f"got shape {arr.shape}."
            )
        if arr.size < 2:
            raise ValueError(f"{name} needs at least 2 observations, got {arr.size}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be finite.")

    m, n = a.size, b.size
    observed = _w1(a, b)

    if cdf is not None:
        return RichResult(
            title="Wasserstein-1 two-sample test",
            payload={
                "statistic": observed,
                "p_value": float(1.0 - cdf(observed)),
                "B": 0,
                "m": int(m),
                "n": int(n),
                "method": "W_1 two-sample test against a supplied null CDF",
            },
        )

    B = int(B)
    if B < 1:
        raise ValueError(f"B must be at least 1, got {B}.")
    pool = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    for i in range(B):
        perm = rng.permutation(pool)
        null[i] = _w1(perm[:m], perm[m:])

    p = (1.0 + float(np.sum(null >= observed))) / (1.0 + B)

    return RichResult(
        title="Wasserstein-1 two-sample test",
        payload={
            "statistic": observed,
            "p_value": p,
            "B": B,
            "m": int(m),
            "n": int(n),
            "null_statistics": null,
            "method": "W_1 two-sample permutation test",
        },
    )


def cheatsheet():
    return "otprm: permutation two-sample test on the 1-Wasserstein distance"
