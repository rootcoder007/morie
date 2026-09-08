# morie.fn -- function file (rootcoder007/morie)
"""Top-k sampling: truncate the softmax to its k largest probabilities."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["top_k_sampling"]


def top_k_sampling(logits, k, temp):
    """Temperature softmax truncated to the ``k`` most probable tokens.

    Top-k is the blunt version of the truncation idea: it removes the
    unreliable tail of the distribution -- where a language model's
    probability mass is mostly noise -- at a FIXED cut, which is its
    weakness as well as its virtue.  When the distribution is peaked, a
    fixed ``k`` keeps tokens that should have been discarded; when it is
    flat, it discards tokens that should have been kept.  Top-p adapts
    the cut to the distribution instead; both are in this package.

    Formula: ``q_i = exp(z_i/T) / sum_j exp(z_j/T)`` restricted to the
    ``k`` largest ``q``, renormalised to sum to one.  Ties are broken by
    the lower index so both language arms keep the same token set.

    Parameters
    ----------
    logits : array-like, shape (V,)
        Unnormalised scores.
    k : int
        Number of tokens to keep; clamped to ``[1, V]``.
    temp : float
        Softmax temperature, strictly positive.

    Returns
    -------
    RichResult
        ``tensor`` (truncated probabilities), ``keep_mask``, ``n_kept``,
        ``entropy`` (of the truncated distribution, in nats), ``k``,
        ``temp``.

    References
    ----------
    Fan, A., Lewis, M. & Dauphin, Y. (2018).  Hierarchical neural story
    generation.  Proceedings of the 56th Annual Meeting of the
    Association for Computational Linguistics, 889-898.
    doi:10.18653/v1/P18-1082.
    """
    z = C.vec(logits)
    V = len(z)
    if V == 0:
        raise ValueError("top_k_sampling: logits must be non-empty")
    temp = float(temp)
    if not temp > 0.0:
        raise ValueError("top_k_sampling: temp must be positive")
    k = int(k)
    if k < 1:
        raise ValueError("top_k_sampling: k must be at least 1")
    k = min(k, V)
    s = [v / temp for v in z]
    mx = max(s)
    e = [math.exp(v - mx) for v in s]
    tot = sum(e)
    probs = [v / tot for v in e]
    order = sorted(range(V), key=lambda i: (-probs[i], i))
    keep = [0.0] * V
    for i in order[:k]:
        keep[i] = 1.0
    filt = [probs[i] if keep[i] > 0.5 else 0.0 for i in range(V)]
    fs = sum(filt)
    filt = [v / fs for v in filt]
    ent = -sum(v * math.log(v) for v in filt if v > 0.0)
    return RichResult(payload={
        "tensor": filt, "keep_mask": keep, "n_kept": float(k),
        "entropy": ent, "k": float(k), "temp": temp,
        "method": "top-k truncated softmax"})


def cheatsheet():
    return "topkS(logits, k, temp): top-k truncated softmax."

# public names resolved by fn/_lazy_map.json
topksampling = top_k_sampling
