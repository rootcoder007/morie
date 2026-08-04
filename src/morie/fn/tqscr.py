# morie.fn -- function file (rootcoder007/morie)
"""QJL attention-score distortion bound."""

import math

from ._richresult import RichResult

__all__ = ["turboquant_score_distortion"]


def turboquant_score_distortion(eps, r, n):
    """Sketch width that keeps every attention score within 1 +/- 3 eps.

    The per-pair bound is not enough on its own -- there are ``n`` keys
    and one bad estimate is enough to move the softmax -- so the score
    bound is the pairwise bound plus a union bound, and the price of
    that union is the ``log n``.  Growing only logarithmically in the
    context length is what makes a fixed bit budget per token viable.

    Formula: if ``max_i ||k_i|| <= r``, ``||q|| <= r`` and
    ``m >= 2 r^2 eps^-2 log n``, then
    ``|Score_hat(i) - Score(i)| <= 3 eps Score(i)`` simultaneously for
    all ``i``, with probability ``1 - 1 / poly(n)``.

    Parameters
    ----------
    eps : float
        Relative score distortion.
    r : float
        Norm bound on the key and query embeddings.
    n : float
        Context length, the number of cached keys.

    Returns
    -------
    RichResult
        ``m_min``, ``m_real``, ``estimate``, ``eps``, ``r``, ``n``.

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
    r = float(r)
    n = float(n)
    m = 2.0 * r * r / (eps * eps) * math.log(n)
    m_min = float(math.ceil(m))
    return RichResult(payload={
        "m_min": m_min, "m_real": m, "estimate": m_min, "eps": eps, "r": r,
        "n": n, "method": "QJL attention-score distortion bound"})


def cheatsheet():
    return "tqscr: QJL attention-score distortion bound."
