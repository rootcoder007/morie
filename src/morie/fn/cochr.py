# morie.fn -- function file (rootcoder007/morie)
"""Cochran's Q test for k≥2 paired binary responses."""

from . import _array_core as np
from . import _stats_core as sp_stats

__all__ = ["cochr"]


def cochr(data, axis=0, cdf=None):
    r"""
    Cochran's Q test for k≥2 paired binary outcomes.

    Tests H0: k treatments have equal effect on binary response
    in a randomized block design.

    Q = (k - 1) [k sum_j G_j^2 - N^2] / [k N - sum_i L_i^2], referred to
    chi-square with k - 1 degrees of freedom, where G_j are treatment
    totals, L_i block totals and N the grand total. For k = 2 it reduces
    to McNemar's uncorrected statistic. When every block responds the
    same way to all treatments the statistic is 0/0 and no block is
    informative; the test then reports Q = 0 and p = 1.

    References
    ----------
    Cochran, W. G. (1950). The comparison of percentages in matched
    samples. Biometrika 37(3/4), 256-266.
    """
    data = np.asarray(data, dtype=int)

    if data.ndim != 2:
        raise ValueError("Input must be 2D (blocks × treatments)")

    b = data.shape[0]  # blocks
    k = data.shape[1]  # treatments

    if b < 2 or k < 2:
        raise ValueError("Need ≥2 blocks and ≥2 treatments")

    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Data must be binary (0/1)")

    # Column totals (treatment successes)
    G = np.sum(data, axis=0)

    # Row totals (block successes)
    L = np.sum(data, axis=1)

    # Cochran's Q (Cochran 1950):
    #   Q = (k - 1) [k sum_j G_j^2 - N^2] / [k N - sum_i L_i^2]
    # with N the grand total. The previous form scaled N^2 by k instead
    # of (k - 1), which is wrong for every input and could even go
    # negative, which Q cannot.
    N = float(np.sum(L))
    num = (k - 1) * (k * float(np.sum(G ** 2)) - N * N)
    den = k * N - float(np.sum(L ** 2))
    if den == 0:
        # Every block responded identically across treatments, so no
        # block carries information about a treatment difference and the
        # treatment totals are necessarily equal: there is no evidence
        # against H0. Q is 0/0 here; report Q = 0, p = 1.
        Q = 0.0
        p_value = 1.0
    else:
        Q = num / den
        # p-value from chi-square with k-1 df
        p_value = float(sp_stats.chi2.sf(Q, k - 1))

    return {
        "statistic": float(Q),
        "p_value": float(p_value),
        "k": int(k),
        "b": int(b),
        "interpretation": "reject" if p_value < 0.05 else "not reject",
    }


def cheatsheet() -> str:
    return "cochr: cochr(data, axis, cdf) -> Cochran's Q test for k≥2 paired binary outcomes."
