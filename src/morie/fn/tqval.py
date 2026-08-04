# morie.fn -- function file (rootcoder007/morie)
"""Token-wise value-cache quantization."""

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["turboquant_value_cache_quantization"]


def turboquant_value_cache_quantization(v, bits=4):
    """Quantize one token value vector by normalise-then-round.

    The value cache does not need the sign trick.  Keys are compared
    against a query by inner product, so their errors are amplified
    through a softmax; values are simply averaged, so plain token-wise
    scaling is enough and the paper says so explicitly.  Doing the
    simple thing here is what leaves the bit budget for the keys, where
    it actually buys accuracy.

    Determinism: half-away-from-zero rounding rather than the language
    ``round``, because Python and R both round half to even but do not
    agree which binary values are exactly half.

    Formula: ``s = max|v|``, ``v_q = round(v / s (2^b - 1))``, and the
    reconstruction is ``v_hat = v_q s / (2^b - 1)``.

    Parameters
    ----------
    v : array-like
        Value embedding for one token.
    bits : int, default 4
        Bits per entry.

    Returns
    -------
    RichResult
        ``v_q``, ``s``, ``v_hat``, ``estimate`` (root mean squared
        reconstruction error), ``bits``, ``d``.

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
    vv = C.vec(v)
    b = int(bits)
    lev = float(2 ** b - 1)
    s = max(abs(t) for t in vv)
    if s <= 0.0:
        s = 1.0
    vq = [S.rnd(t / s * lev) for t in vv]
    vhat = [t * s / lev for t in vq]
    err = sum((vv[i] - vhat[i]) ** 2 for i in range(len(vv))) / len(vv)
    return RichResult(payload={
        "v_q": vq, "s": s, "v_hat": vhat, "estimate": err ** 0.5, "bits": b,
        "d": len(vv), "method": "Token-wise value-cache quantization"})


def cheatsheet():
    return "tqval: Token-wise value-cache quantization."
