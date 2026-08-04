# morie.fn -- function file (rootcoder007/morie)
"""Orthogonalized JL sketch matrix."""

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["turboquant_orthogonalized_jl"]


def turboquant_orthogonalized_jl(S_mat):
    """Replace a Gaussian sketch by one with orthonormal rows.

    Independent Gaussian rows are wasteful: they partly repeat one
    another, so some of the sketch width buys nothing.  Orthogonalising
    the rows removes that redundancy and, in the paper measurements,
    almost always improves the quantizer -- the same reason a
    subsampled orthogonal transform beats a plain Gaussian projection
    at equal width.

    Determinism: modified Gram-Schmidt on the transpose, with a
    non-negative ``R`` diagonal, so there is no sign convention to
    disagree about.

    Formula: ``Q, R = QR(S^T)``; return ``S_orth = Q^T``, whose rows are
    orthonormal.

    Parameters
    ----------
    S_mat : array-like, shape (m, d)
        Gaussian sketch matrix with ``m <= d``.

    Returns
    -------
    RichResult
        ``S_orth``, ``estimate`` (``S_orth[0][0]``), ``m``, ``d``,
        ``orth_err``.

    References
    ----------
    Zandieh, A., Daliri, M. & Han, I. (2024).  QJL: 1-bit quantized
    JL transform for KV cache quantization with zero overhead.
    arXiv:2406.03482.  Fetched and read; the definitions and bounds used
    here are that paper own (definition 3.1, fact 3.4, lemma 3.5,
    theorem 3.6).  The KV-cache system built on it is Zandieh, A. et al.
    (2025), TurboQuant: online vector quantization with near-optimal
    distortion rate, arXiv:2504.19874.
    """
    Sm = C.mat(S_mat)
    m, d = C.shape(Sm)
    Q, _ = S.qr_mgs(C.transpose(Sm))
    So = C.transpose(Q)
    err = 0.0
    for i in range(m):
        for j in range(m):
            v = sum(So[i][t] * So[j][t] for t in range(d)) - (1.0 if i == j else 0.0)
            if abs(v) > err:
                err = abs(v)
    return RichResult(payload={
        "S_orth": So, "estimate": So[0][0], "m": m, "d": d, "orth_err": err,
        "method": "Orthogonalized JL sketch matrix"})


def cheatsheet():
    return "tqorth: Orthogonalized JL sketch matrix."
