# morie.fn -- function file (rootcoder007/morie)
"""Exact null distribution of Spearman's coefficient."""

from itertools import permutations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["gibbons_spearman_exact"]


def gibbons_spearman_exact(n, rho=None):
    r"""Exact permutation null distribution of Spearman's r_s.

    Enumerates all n! rankings against a fixed reference and computes
    :math:`r_s = 1 - 6\sum d_i^2/(n^3 - n)` for each (Gibbons
    Ch. 11.3). Feasible for n <= 8; larger n raises. The exact null
    variance equals 1/(n-1), which anchors the asymptotic module.

    Parameters
    ----------
    n : int, 2..8
        Number of objects.
    rho : float, optional
        Observed r_s; when given, exact tail probabilities at rho are
        returned.

    Returns
    -------
    RichResult
        keys: ``support``, ``pmf``, ``mean`` (0), ``var`` (1/(n-1)),
        ``p_ge``/``p_two`` (if rho given), ``n``, ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Ch. 11.3.
    """
    n = int(n)
    if not 2 <= n <= 8:
        raise ValueError(
            f"exact enumeration is limited to 2 <= n <= 8, got {n}."
        )
    ref = np.arange(1, n + 1)
    denom = float(n**3 - n)
    counts = {}
    for perm in permutations(range(1, n + 1)):
        d2 = int(np.sum((np.array(perm) - ref) ** 2))
        counts[d2] = counts.get(d2, 0) + 1
    total = float(np.prod(range(1, n + 1)))
    d2s = np.array(sorted(counts))
    support = 1.0 - 6.0 * d2s / denom
    order = np.argsort(support)
    support = support[order]
    pmf = np.array([counts[d] for d in d2s])[order] / total
    mean = float(np.sum(support * pmf))
    var = float(np.sum(support**2 * pmf) - mean**2)
    payload = {
        "support": support, "pmf": pmf, "mean": mean, "var": var,
        "n": n, "method": "Exact Spearman null distribution by enumeration",
    }
    if rho is not None:
        rho = float(rho)
        payload["p_ge"] = float(np.sum(pmf[support >= rho - 1e-12]))
        payload["p_two"] = float(min(1.0, np.sum(pmf[np.abs(support) >= abs(rho) - 1e-12])))
    return RichResult(payload=payload)


def cheatsheet():
    return "gb_sp2: enumerate n! rankings; exact var = 1/(n-1)"
