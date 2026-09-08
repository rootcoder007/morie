# morie.fn -- function file (rootcoder007/morie)
"""MuZero categorical value head and its invertible scaling."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["mzvalue", "muzero_predict_value"]


def mzvalue(logits, support=300, epsilon=0.001):
    """Scalar value from the categorical head of the prediction function.

    The prediction function returns p, v = f(s).  In Atari the value and
    reward heads are categorical over a discrete support of the integers
    -support..support, and the targets are first squashed by the
    invertible transform of Pohlen et al.,

        h(x)      = sign(x) ( sqrt(|x| + 1) - 1 + eps x )
        h^{-1}(y) = sign(y) [ ( (sqrt(1 + 4 eps (|y| + 1 + eps)) - 1)
                                / (2 eps) )^2 - 1 ]

    with eps = 0.001.  A scalar is recovered from the categorical head by
    taking the expectation over the support and then inverting h.

    Parameters
    ----------
    logits : array-like
        Head outputs over the support, length 2*support + 1.  They are
        turned into probabilities by a softmax; pass log-probabilities or
        logits interchangeably.
    support : int
        Half-width of the integer support.
    epsilon : float
        eps in the transform.

    Returns
    -------
    RichResult
        ``value``, ``expected``, ``prob``, ``support``, ``epsilon``, ``k``.

    References
    ----------
    Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K.,
    Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D.,
    Graepel, T., Lillicrap, T. and Silver, D. (2020), "Mastering Atari,
    Go, chess and shogi by planning with a learned model", Nature 588,
    604-609; arXiv:1911.08265.  Read from the ar5iv rendering of the
    arXiv source.  Appendix F gives h(x) = sign(x)(sqrt(|x|+1) - 1 + eps x) with
    eps = 0.001, the support of 601 integers from -300 to 300, and the
    recovery x = x_low p_low + x_high p_high from the two adjacent
    supports, which the expectation over the whole support generalises.
    """
    z = C.vec(logits)
    s = int(support)
    if len(z) != 2 * s + 1:
        raise ValueError("logits must have length 2*support + 1")
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("epsilon must be strictly positive")
    m = max(z)
    e = [math.exp(v - m) for v in z]
    tot = sum(e)
    p = [v / tot for v in e]
    y = sum(p[i] * (i - s) for i in range(2 * s + 1))
    sg = 1.0 if y >= 0.0 else -1.0
    a = (math.sqrt(1.0 + 4.0 * eps * (abs(y) + 1.0 + eps)) - 1.0) / (2.0 * eps)
    return RichResult(payload={
        "value": sg * (a * a - 1.0), "expected": y, "prob": p,
        "support": s, "epsilon": eps, "k": 2 * s + 1,
        "method": "MuZero categorical value head (Schrittwieser et al. 2020 App. F)"})


muzero_predict_value = mzvalue


def cheatsheet():
    return "agmupv: MuZero categorical value head and its invertible scaling."
