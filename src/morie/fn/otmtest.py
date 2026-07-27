# morie.fn -- function file (rootcoder007/morie)
"""Kernel two-sample test using the maximum mean discrepancy."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["ot_mmd_two_sample"]

_KERNELS = ("rbf", "laplacian", "linear")


def _as2d(a, name):
    A = np.asarray(a, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got shape {A.shape}.")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} must be finite.")
    return A


def _gram(A, B, kernel, gamma):
    if kernel == "linear":
        return A @ B.T
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=-1)
    if kernel == "rbf":
        return np.exp(-gamma * d2)
    return np.exp(-gamma * np.sqrt(d2))  # laplacian


def _median_gamma(Z):
    """Median heuristic: gamma = 1 / median squared pairwise distance."""
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1)
    med = float(np.median(d2[np.triu_indices(Z.shape[0], k=1)])) if Z.shape[0] > 1 else 0.0
    return 1.0 / med if med > 0 else 1.0


def ot_mmd_two_sample(X, Y, kernel="rbf", B=999, cdf=None, gamma=None, unbiased=False, seed=None):
    r"""Kernel two-sample test (Gretton et al. 2012).

    The maximum mean discrepancy is the largest difference in expectation
    over the unit ball of a reproducing-kernel Hilbert space. Its biased
    empirical estimate keeps the diagonal terms,

    .. math::

        \mathrm{MMD}_b^2 = \frac{1}{m^2}\sum_{i,j} k(x_i,x_j)
                         + \frac{1}{n^2}\sum_{i,j} k(y_i,y_j)
                         - \frac{2}{mn}\sum_{i,j} k(x_i,y_j)

    while the unbiased estimate of their equation (3) drops them:

    .. math::

        \mathrm{MMD}_u^2 = \frac{1}{m(m-1)}\sum_{i \ne j} k(x_i,x_j)
                         + \frac{1}{n(n-1)}\sum_{i \ne j} k(y_i,y_j)
                         - \frac{2}{mn}\sum_{i,j} k(x_i,y_j)

    The null distribution has no simple closed form, so the p-value comes
    from permuting the group labels: pool the two samples, re-split at
    the original sizes ``B`` times, and rank the observed statistic among
    the permuted ones. Any bias in the estimator is shared by the
    observed and permuted values, so it cancels in the ranking -- which
    is why the biased form is a perfectly good test statistic even though
    the unbiased one is the better *estimate* of MMD^2.

    Parameters
    ----------
    X, Y : array-like
        The two samples, ``(m, d)`` and ``(n, d)``. One-dimensional input
        is read as a column. The feature dimension must match.
    kernel : {"rbf", "laplacian", "linear"}, default "rbf"
        Kernel k. A characteristic kernel (rbf, laplacian) makes MMD zero
        only when the distributions are equal; the linear kernel compares
        means alone and is blind to any difference that leaves the mean
        unchanged.
    B : int, default 999
        Number of label permutations.
    cdf : callable, optional
        Null CDF of the statistic, replacing the permutation null.
    gamma : float, optional
        Kernel bandwidth. Defaults to the median heuristic on the pooled
        sample, ``1 / median squared pairwise distance``. Ignored by the
        linear kernel.
    unbiased : bool, default False
        Use equation (3) rather than the biased estimate. The biased form
        is the default because it is what this module has always
        documented, and because it is non-negative.
    seed : int, optional
        Seed for the permutations.

    Returns
    -------
    RichResult
        keys: ``statistic`` (MMD^2), ``p_value``, ``kernel``, ``gamma``,
        ``B``, ``m``, ``n``, ``unbiased``, ``null_statistics``,
        ``method``.

    References
    ----------
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Scholkopf, B. & Smola,
    A. (2012). A kernel two-sample test. *Journal of Machine Learning
    Research*, 13, 723-773.
    """
    A = _as2d(X, "X")
    Bm = _as2d(Y, "Y")
    if A.shape[1] != Bm.shape[1]:
        raise ValueError(f"X and Y must share a feature dimension; got {A.shape[1]} and {Bm.shape[1]}.")
    m, n = A.shape[0], Bm.shape[0]
    if m < 2 or n < 2:
        raise ValueError(f"Both samples need at least 2 observations, got m={m}, n={n}.")
    if kernel not in _KERNELS:
        raise ValueError(f"kernel must be one of {_KERNELS}, got {kernel!r}.")

    Z = np.vstack([A, Bm])
    g = _median_gamma(Z) if gamma is None else float(gamma)
    if g <= 0:
        raise ValueError(f"gamma must be positive, got {g}.")
    K = _gram(Z, Z, kernel, g)

    def mmd2(idx_x, idx_y):
        Kxx = K[np.ix_(idx_x, idx_x)]
        Kyy = K[np.ix_(idx_y, idx_y)]
        Kxy = K[np.ix_(idx_x, idx_y)]
        mm, nn = idx_x.size, idx_y.size
        if unbiased:
            sxx = (Kxx.sum() - np.trace(Kxx)) / (mm * (mm - 1))
            syy = (Kyy.sum() - np.trace(Kyy)) / (nn * (nn - 1))
        else:
            sxx = Kxx.sum() / (mm * mm)
            syy = Kyy.sum() / (nn * nn)
        return float(sxx + syy - 2.0 * Kxy.sum() / (mm * nn))

    ix = np.arange(m)
    iy = np.arange(m, m + n)
    observed = mmd2(ix, iy)

    if cdf is not None:
        return RichResult(
            title="Kernel two-sample test (MMD)",
            payload={
                "statistic": observed,
                "p_value": float(1.0 - cdf(observed)),
                "kernel": kernel,
                "gamma": g,
                "B": 0,
                "m": int(m),
                "n": int(n),
                "unbiased": bool(unbiased),
                "method": "MMD two-sample test against a supplied null CDF",
            },
        )

    B = int(B)
    if B < 1:
        raise ValueError(f"B must be at least 1, got {B}.")
    rng = np.random.default_rng(seed)
    null = np.empty(B)
    pool = np.arange(m + n)
    for b in range(B):
        perm = rng.permutation(pool)
        null[b] = mmd2(perm[:m], perm[m:])

    p = (1.0 + float(np.sum(null >= observed))) / (1.0 + B)

    return RichResult(
        title="Kernel two-sample test (MMD)",
        payload={
            "statistic": observed,
            "p_value": p,
            "kernel": kernel,
            "gamma": g,
            "B": B,
            "m": int(m),
            "n": int(n),
            "unbiased": bool(unbiased),
            "null_statistics": null,
            "method": "MMD two-sample permutation test (Gretton et al. 2012)",
        },
    )


def cheatsheet():
    return "otmtest: kernel two-sample test via maximum mean discrepancy"
