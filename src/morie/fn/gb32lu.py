# morie.fn -- function file (rootcoder007/morie)
"""Exact distribution of runs up and down by recursion."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_runs_up_down_recur"]


def _ud_counts_exact(n):
    """Brute-force enumeration over all n! orderings -- exact for n <= 9.

    Enumeration is deliberately the engine here rather than the
    Ch. 3.4 insertion recurrence: the recurrence's transition counts
    are easy to get subtly wrong, and at these n the exact count is
    cheap. The moments (2n-1)/3 and (16n-29)/90 are asserted against
    this pmf in the tests.
    """
    from itertools import permutations

    agg = {}
    for p in permutations(range(n)):
        d = np.sign(np.diff(p))
        r = 1 + int(np.sum(d[1:] != d[:-1]))
        agg[r] = agg.get(r, 0) + 1
    return agg


def gibbons_runs_up_down_recur(x=None, n=None):
    r"""Exact null distribution of the runs up-and-down count.

    For a sequence of n distinct exchangeable values, the number of
    maximal monotone runs R_ud has mean (2n - 1)/3 and variance
    (16n - 29)/90 (Gibbons Ch. 3.4). This returns the exact pmf for
    n <= 9 by enumeration of all n! orderings and, when a data vector
    is supplied, the observed count with its exact tail probability.

    Parameters
    ----------
    x : array-like, optional
        Observed sequence (distinct values); its R_ud is scored.
    n : int, optional
        Length; required when x is omitted. Must satisfy 3 <= n <= 9
        for the exact route.

    Returns
    -------
    RichResult
        keys: ``support``, ``pmf``, ``mean``, ``var``, ``observed``/
        ``p_le``/``p_ge`` (if x given), ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 3.4.
    """
    obs = None
    if x is not None:
        x = np.asarray(x, dtype=float).ravel()
        if np.unique(x).size != x.size:
            raise ValueError("runs up/down need distinct values (no ties).")
        n = x.size
        d = np.sign(np.diff(x))
        obs = 1 + int(np.sum(d[1:] != d[:-1]))
    if n is None:
        raise ValueError("supply either x or n.")
    n = int(n)
    if not 3 <= n <= 9:
        raise ValueError(
            f"exact enumeration is limited to 3 <= n <= 9, got {n}; "
            "use the normal approximation with the Ch. 3.4 moments."
        )
    import math

    agg = _ud_counts_exact(n)
    support = np.array(sorted(agg))
    total = float(math.factorial(n))
    pmf = np.array([agg[r] for r in support]) / total
    mean = float(np.sum(support * pmf))
    var = float(np.sum(support**2 * pmf) - mean**2)
    payload = {
        "support": support.astype(int), "pmf": pmf, "mean": mean, "var": var,
        "n": n, "method": "Exact runs up/down pmf by enumeration (Gibbons Ch. 3.4)",
    }
    if obs is not None:
        payload["observed"] = obs
        payload["p_le"] = float(np.sum(pmf[support <= obs]))
        payload["p_ge"] = float(np.sum(pmf[support >= obs]))
    return RichResult(payload=payload)


def cheatsheet():
    return "gb32lu: exact R_ud pmf n<=9; mean (2n-1)/3 checked against it"
