# morie.fn -- function file (rootcoder007/morie)
"""QJL online key-cache quantizer (algorithm 1)."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["turboquant_online_key_quantizer"]


def turboquant_online_key_quantizer(k, S_mat, q=None):
    """Store one key as a sign vector plus its norm.

    Only the norm survives the sign quantization, so it is kept
    alongside -- that pair is the whole cache entry.  Because the
    estimator is asymmetric (the query is projected but not quantized)
    the inner product estimate stays unbiased; quantizing both sides
    would give an unbiased *angle* and then a biased inner product once
    the cosine is applied, which is the mistake the asymmetry avoids.

    Formula: store ``k_tilde = sign(S k)`` and ``nu = ||k||_2``; the
    estimator is
    ``Prod(q, k) = (sqrt(pi / 2) / m) nu <S q, k_tilde>``.

    Parameters
    ----------
    k : array-like, shape (d,)
        Key embedding to cache.
    S_mat : array-like, shape (m, d)
        JL sketch matrix.
    q : array-like, optional
        Query embedding; when given, the unbiased inner-product estimate
        is returned as ``estimate``.

    Returns
    -------
    RichResult
        ``k_tilde``, ``nu``, ``m``, ``d``, and ``estimate`` (the
        estimated inner product, or ``nu`` when no query is supplied).

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
    m = len(Sm)
    ktil = [S.sgn(v) for v in C.matvec(Sm, kv)]
    nu = math.sqrt(sum(v * v for v in kv))
    est = nu
    if q is not None:
        sq = C.matvec(Sm, C.vec(q))
        est = math.sqrt(math.pi / 2.0) / m * nu * sum(sq[i] * ktil[i] for i in range(m))
    return RichResult(payload={
        "k_tilde": ktil, "nu": nu, "m": m, "d": len(kv), "estimate": est,
        "method": "QJL online key quantizer with unbiased inner product"})


def cheatsheet():
    return "tqalg1: QJL online key-cache quantizer (algorithm 1)."
