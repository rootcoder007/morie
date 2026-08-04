# morie.fn -- function file (rootcoder007/morie)
"""QJL inner-product distortion bound."""

import math

from ._richresult import RichResult

__all__ = ["turboquant_distortion_bound"]


def turboquant_distortion_bound(eps, delta):
    """Smallest sketch size meeting an inner-product distortion target.

    The bound is what makes the scheme predictable rather than merely
    empirical: it says how wide the sketch must be, and the answer does
    not mention the embedding dimension at all.  Cost per token is
    therefore fixed once the accuracy target is fixed, which is why a
    long context does not cost more per key.

    Formula: with ``m >= (4 / 3) (1 + eps) / eps^2 log(2 / delta)``,
    ``Pr[|Prod(q, k) - <q, k>| > eps ||q|| ||k||] <= delta``.

    Parameters
    ----------
    eps : float
        Relative distortion target.
    delta : float
        Failure probability.

    Returns
    -------
    RichResult
        ``m_min`` (the smallest integer m satisfying the bound),
        ``m_real`` (the bound before rounding up), ``estimate``,
        ``eps``, ``delta``.

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
    eps = float(eps)
    delta = float(delta)
    m = (4.0 / 3.0) * (1.0 + eps) / (eps * eps) * math.log(2.0 / delta)
    m_min = float(math.ceil(m))
    return RichResult(payload={
        "m_min": m_min, "m_real": m, "estimate": m_min, "eps": eps,
        "delta": delta, "method": "QJL inner-product distortion bound"})


def cheatsheet():
    return "tqdist: QJL inner-product distortion bound."
