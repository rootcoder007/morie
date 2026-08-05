# morie.fn -- function file (rootcoder007/morie)
"""Locally typical sampling: keep tokens whose surprisal is near the entropy."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["typical_sampling"]


def typical_sampling(logits, tau):
    """Truncate to the locally typical set at cumulative mass ``tau``.

    Top-k and top-p both cut from the TOP of the distribution, which
    quietly assumes the most probable token is always the most
    appropriate one.  Typical decoding drops that assumption: it ranks
    tokens by how close their surprisal ``-log q_i`` is to the
    conditional entropy ``H(q)`` and keeps the smallest such set with
    cumulative probability at least ``tau``.  The very top of the
    distribution can therefore be excluded when it is far more probable
    than the entropy says it should be, which is the mechanism behind
    the reduction in degenerate repetition.

    Formula: ``H = -sum_i q_i log q_i``, rank by ``|-log q_i - H|``
    ascending (ties by lower index), accumulate ``q`` in that order and
    stop at the first index whose running total reaches ``tau``.

    Parameters
    ----------
    logits : array-like, shape (V,)
        Unnormalised scores.  No temperature is applied; scale the
        logits yourself if you want one.
    tau : float
        Cumulative-probability target in ``(0, 1]``.

    Returns
    -------
    RichResult
        ``tensor`` (truncated probabilities), ``keep_mask``, ``n_kept``,
        ``entropy`` (of the full distribution, in nats), ``tau``.

    References
    ----------
    Meister, C., Pimentel, T., Wiher, G. & Cotterell, R. (2023).
    Locally typical sampling.  Transactions of the Association for
    Computational Linguistics 11:102-121.  doi:10.1162/tacl_a_00536.
    """
    z = C.vec(logits)
    V = len(z)
    if V == 0:
        raise ValueError("typical_sampling: logits must be non-empty")
    tau = float(tau)
    if not (0.0 < tau <= 1.0):
        raise ValueError("typical_sampling: tau must be in (0, 1]")
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    tot = sum(e)
    probs = [v / tot for v in e]
    ent = -sum(v * math.log(v) for v in probs if v > 0.0)
    dev = [abs(-math.log(p) - ent) if p > 0.0 else float("inf") for p in probs]
    order = sorted(range(V), key=lambda i: (dev[i], i))
    keep = [0.0] * V
    run = 0.0
    nk = 0
    for i in order:
        keep[i] = 1.0
        run += probs[i]
        nk += 1
        if run >= tau:
            break
    filt = [probs[i] if keep[i] > 0.5 else 0.0 for i in range(V)]
    fs = sum(filt)
    filt = [v / fs for v in filt]
    return RichResult(payload={
        "tensor": filt, "keep_mask": keep, "n_kept": float(nk),
        "entropy": ent, "tau": tau,
        "method": "locally typical truncated softmax"})


def cheatsheet():
    return "ttypc(logits, tau): locally typical sampling truncation."
