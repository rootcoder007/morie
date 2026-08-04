# morie.fn -- function file (rootcoder007/morie)
"""QJL sign quantizer."""

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["turboquant_qjl_sign_quantizer"]


def turboquant_qjl_sign_quantizer(k, S_mat):
    """Quantize a key embedding to one bit per sketch row.

    The trick is that the sign of a random projection keeps enough
    information to estimate an inner product, so the stored key costs
    one bit per row of the sketch and nothing else -- no per-group
    scale, no zero point, no grouping at all.  That is what makes the
    overhead zero: there is no side information to store.

    Formula: ``H_S(k) := sign(S k)``, with ``S`` an ``m by d`` matrix of
    i.i.d. standard normals.  Zero maps to ``+1`` so the range really is
    ``{-1, +1}^m``.

    Parameters
    ----------
    k : array-like, shape (d,)
        Key embedding.
    S_mat : array-like, shape (m, d)
        JL sketch matrix.

    Returns
    -------
    RichResult
        ``signs``, ``m``, ``d``, and ``estimate`` (the mean sign, a
        one-number summary of the sketch balance).

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
    kv = C.vec(k)
    Sm = C.mat(S_mat)
    proj = C.matvec(Sm, kv)
    signs = [S.sgn(v) for v in proj]
    return RichResult(payload={
        "signs": signs, "m": len(signs), "d": len(kv),
        "estimate": sum(signs) / len(signs),
        "method": "QJL sign quantizer H_S(k) = sign(S k)"})


def cheatsheet():
    return "tqhs: QJL sign quantizer."
