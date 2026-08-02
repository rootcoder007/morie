# morie.fn -- function file (rootcoder007/morie)
"""Fork (common cause) structure A<-B->C: B is confounder."""

from . import _array_core as np
from . import _stats_core as stats

from ._richresult import RichResult

__all__ = ["fork_structure"]


def _fisher_z(r, n, k):
    if n - k - 3 <= 0:
        raise ValueError("too few observations for the Fisher-z test.")
    r = min(max(r, -0.999999), 0.999999)  # degenerate |r| = 1 fixtures
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - k - 3)
    return float(2 * stats.norm.sf(abs(z)))


def _pcorr(a, b, c=None):
    n = a.size
    D = np.column_stack([np.ones(n)] + ([c[:, None]] if c is not None else []))
    ra = a - D @ np.linalg.lstsq(D, a, rcond=None)[0]
    rb = b - D @ np.linalg.lstsq(D, b, rcond=None)[0]
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def fork_structure(A, B, C, alpha=0.01):
    r"""Test the independence signature of a fork A <- B -> C.

    A fork implies marginal dependence between A and C (the common
    cause B confounds them) and conditional independence given B:

    .. math:: A \not\perp C, \qquad A \perp C \mid B.

    Both are tested by Fisher-z (partial) correlation. The same
    signature holds for a chain, so passing is consistent with a fork,
    not proof of one -- forks and chains are Markov equivalent.

    Parameters
    ----------
    A, B, C : array-like, shape (n,)
        Observations of the three variables.
    alpha : float, default 0.01
        Test level.

    Returns
    -------
    RichResult
        keys: ``marginal_corr``, ``marginal_p``, ``partial_corr``,
        ``partial_p``, ``marginally_dependent``,
        ``conditionally_independent``, ``consistent_with_fork``,
        ``n``, ``method``.

    References
    ----------
    Pearl, J. (2009). *Causality* (2nd ed.). Cambridge University
    Press. Sec. 1.2.3 (d-separation: chains, forks, colliders).
    """
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    C = np.asarray(C, dtype=float).ravel()
    n = A.size
    if not (B.size == n and C.size == n):
        raise ValueError("A, B, C must have equal length.")

    r_m = _pcorr(A, C)
    p_m = _fisher_z(r_m, n, 0)
    r_p = _pcorr(A, C, B)
    p_p = _fisher_z(r_p, n, 1)

    dep = p_m < alpha
    ind = p_p >= alpha
    return RichResult(
        payload={
            "marginal_corr": r_m,
            "marginal_p": p_m,
            "partial_corr": r_p,
            "partial_p": p_p,
            "marginally_dependent": bool(dep),
            "conditionally_independent": bool(ind),
            "consistent_with_fork": bool(dep and ind),
            "n": int(n),
            "method": "Fork independence signature: A ~ C marginally, A _||_ C | B",
        }
    )


def cheatsheet():
    return "frkst: fork A<-B->C -- dependent marginally, independent given B (Fisher-z)"
