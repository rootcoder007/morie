# morie.fn -- function file (rootcoder007/morie)
"""Eigengap heuristic for the number of clusters."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["eigengap", "sgt_eigengap_heuristic"]


def eigengap(values, kmax=None):
    """Choose k from the spectrum: the gap after the small eigenvalues.

    The heuristic is stated the right way round here.  The goal is to
    choose k so that lambda_1, ..., lambda_k are all SMALL and
    lambda_{k+1} is relatively LARGE, which is not the same as taking
    the largest gap anywhere in the spectrum -- a large gap high up the
    spectrum says nothing about cluster count.  ``k`` is therefore the
    argmax of the gap, and ``gaps`` is returned in full so a caller can
    see whether the choice was clear-cut or arbitrary.

    In the ideal case of k completely disconnected clusters the
    eigenvalue 0 has multiplicity k and the gap to lambda_{k+1} is the
    whole story; on real data it is a heuristic and nothing more.

    Formula: gamma_k = |lambda_k - lambda_{k+1}|;  khat = argmax_k gamma_k

    Parameters
    ----------
    values : array-like
        Laplacian eigenvalues in INCREASING order.
    kmax : int, optional
        Largest k considered (default: len(values) - 1).

    Returns
    -------
    RichResult
        ``k``, ``gap``, ``gaps``, ``values``, ``n_zero`` (eigenvalues
        below 1e-10, the ideal-case cluster count), ``kmax``.

    References
    ----------
    von Luxburg (2007), A Tutorial on Spectral Clustering, Statistics
    and Computing 17(4), 395-416, Section 8.3: "the goal is to choose
    the number k such that all eigenvalues lambda_1, ..., lambda_k are
    very small, but lambda_{k+1} is relatively large", justified by
    perturbation theory since "in the ideal case of k completely
    disconnected clusters, the eigenvalue 0 has multiplicity k, and
    then there is a gap to the (k+1)th eigenvalue"; the spectral gap is
    written gamma_k = |lambda_k - lambda_{k+1}|.  Fetched from
    arXiv:0711.0189.
    """
    v = C.vec(values)
    n = len(v)
    if n < 2:
        raise ValueError("at least two eigenvalues are required")
    if any(v[i] > v[i + 1] + 1e-12 for i in range(n - 1)):
        raise ValueError("eigenvalues must be given in increasing order")
    km = n - 1 if kmax is None else int(kmax)
    if not 1 <= km <= n - 1:
        raise ValueError("kmax must satisfy 1 <= kmax <= len(values) - 1")
    gaps = [abs(v[k] - v[k - 1]) for k in range(1, km + 1)]
    best = max(range(km), key=lambda i: (gaps[i], -i))
    return RichResult(payload={
        "k": float(best + 1), "gap": gaps[best], "gaps": gaps,
        "values": v, "n_zero": float(sum(1 for x in v if x < 1e-10)),
        "kmax": float(km),
        "method": "Eigengap heuristic, von Luxburg (2007) Section 8.3"})


sgt_eigengap_heuristic = eigengap


def cheatsheet():
    return "sgtegap: gamma_k = |lambda_k - lambda_{k+1}|; k = argmax gamma_k"
