# morie.fn -- function file (rootcoder007/morie)
"""Exact null distribution of Kendall's tau."""

from itertools import permutations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_kendall_exact"]


def gibbons_kendall_exact(n, t=None):
    r"""Exact permutation null distribution of Kendall's T.

    Enumerates all n! rankings, counts concordant-minus-discordant
    for each, and tabulates the distribution of
    :math:`T = (P - Q)/\binom{n}{2}` under the null that all rankings
    are equally likely (Gibbons Ch. 11.2). Feasible for n <= 8 (8! =
    40320); larger n raises rather than silently switching to an
    approximation -- the normal route lives in the asymptotic module.

    Parameters
    ----------
    n : int, 2..8
        Number of objects.
    t : float, optional
        A tau value; when given, the exact one- and two-sided
        p-values at t are returned.

    Returns
    -------
    RichResult
        keys: ``support`` (possible tau values), ``pmf``, ``mean``
        (0), ``var`` (matches 2(2n+5)/(9n(n-1))), ``p_ge``/``p_two``
        (if t given), ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 11.2.
    """
    n = int(n)
    if not 2 <= n <= 8:
        raise ValueError(
            f"exact enumeration is limited to 2 <= n <= 8, got {n}; "
            "use the asymptotic normal for larger n."
        )
    npairs = n * (n - 1) // 2
    counts = {}
    for perm in permutations(range(n)):
        p = np.array(perm)
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += 1 if (p[j] - p[i]) > 0 else -1
        counts[s] = counts.get(s, 0) + 1
    import math

    total = float(math.factorial(n))
    ss = np.array(sorted(counts))
    support = ss / npairs
    pmf = np.array([counts[s] for s in ss]) / total
    mean = float(np.sum(support * pmf))
    var = float(np.sum(support**2 * pmf) - mean**2)
    payload = {
        "support": support, "pmf": pmf, "mean": mean, "var": var,
        "n": n, "method": "Exact Kendall tau null distribution by enumeration",
    }
    if t is not None:
        t = float(t)
        payload["p_ge"] = float(np.sum(pmf[support >= t - 1e-12]))
        payload["p_two"] = float(min(1.0, np.sum(pmf[np.abs(support) >= abs(t) - 1e-12])))
    return RichResult(payload=payload)


def cheatsheet():
    return "gb_kt2: enumerate n! rankings; var check = 2(2n+5)/(9n(n-1))"
