# morie.fn -- function file (rootcoder007/morie)
"""Nystrom low-rank approximation of a kernel matrix."""

from . import _gp_core as G

from ._richresult import RichResult

__all__ = ['nystromap', 'nystrom_approximation']


def nystromap(X, m_index, kernel='linear', gamma=None):
    """Nystrom low-rank approximation of a kernel matrix.

    Formula: Q = K_nm * K_mm^- * K_nm',  m the retained subset of records

    Parameters
    ----------
    X : array-like, shape (n, p)
        One record per row.
    m_index : array-like of int
        1-based row indices of the retained subset.
    kernel : str
        Kernel name: linear, gaussian, polynomial, exponential or sigmoid.
    gamma : float or None
        Kernel bandwidth; None uses 1/p.

    Returns
    -------
    RichResult
        ``Q``, ``m``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 8: the Nystrom method for compressing a kernel matrix onto a retained subset of records; the implementation delegates to the chapter-8 Nystrom routine already verified against the book for this shelf.  Read from the chapter PDF, not recalled.
    """
    idx = [int(v) - 1 for v in m_index]
    if any(i < 0 for i in idx):
        raise ValueError("m_index is 1-based")
    out = G.nystrom_kernel(X, idx, kernel=kernel, gamma=gamma)
    Q = out["Q"] if isinstance(out, dict) else out
    return RichResult(payload={
        "Q": Q, "m": len(idx), "n": len(Q),
        "method": "Nystrom kernel approximation, MVSML Chap. 8"})


nystrom_approximation = nystromap


def cheatsheet():
    return 'nystm: Nystrom low-rank approximation of a kernel matrix.'
