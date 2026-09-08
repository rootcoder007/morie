# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Top-k sampling: renormalize over the k most likely tokens."""

from . import _array_core as np

from ._richresult import RichResult
from .grn021 import softmax_vector

__all__ = ["geron_topk_sampling"]

_METHOD = "Top-k truncated sampling distribution"


def geron_topk_sampling(logits, k, T=1.0):
    r"""Keep the ``k`` best tokens and renormalise; zero the rest.

    .. math::
        p'_i = \begin{cases}
            p_i / \sum_{j \in \mathrm{TopK}} p_j & i \in \mathrm{TopK}\\
            0 & \text{otherwise}\end{cases}

    The tail of a vocabulary distribution is enormous -- tens of
    thousands of tokens each with tiny probability, whose *sum* is not
    tiny.  Sampling from the full distribution therefore picks
    nonsense at a noticeable rate; truncation removes that mass entirely
    rather than merely discouraging it.  ``k`` is a fixed count, so on a
    flat distribution it cuts too much and on a peaked one too little,
    which is exactly the complaint nucleus (top-p) sampling answers.
    Ties at the boundary are broken by index so the result is
    deterministic.

    Parameters
    ----------
    logits : array-like, shape (V,)
    k : int
        Tokens to keep, ``1 <= k <= V``.
    T : float, optional
        Temperature applied before truncation.

    Returns
    -------
    RichResult
        Payload keys ``probabilities`` (length V, zeros outside the top
        k), ``kept_indices``, ``kept_mass`` (before renormalising),
        ``entropy``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 15, Sampling strategies section.

    Examples
    --------
    Logits ``[2, 1, 0]``, ``k = 2``: the kept mass is
    ``0.665241 + 0.244728 = 0.909969`` and the survivors are rescaled by
    it.

    >>> r = geron_topk_sampling([2.0, 1.0, 0.0], k=2)
    >>> r["kept_indices"]
    [0, 1]
    >>> [round(p, 6) for p in r["probabilities"]]
    [0.731059, 0.268941, 0.0]
    >>> round(r["kept_mass"], 6)
    0.909969

    ``k = 1`` is greedy decoding:

    >>> geron_topk_sampling([2.0, 1.0, 0.0], k=1)["probabilities"]
    [1.0, 0.0, 0.0]
    """
    z = np.asarray(logits, dtype=float).ravel()
    if z.size == 0:
        raise ValueError("logits is empty.")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits contains non-finite values.")
    k = int(k)
    if not (1 <= k <= z.size):
        raise ValueError(f"k must lie in [1, {z.size}], got {k}.")
    T = float(T)
    if not np.isfinite(T) or T <= 0:
        raise ValueError(f"T must be strictly positive, got {T}.")

    p = softmax_vector(z / T)
    order = np.lexsort((np.arange(z.size), -p))     # ties -> lower index first
    keep = np.sort(order[:k])
    mass = float(p[keep].sum())
    if mass <= 0:
        raise ValueError("the top-k tokens carry no probability mass; check the logits.")
    out = np.zeros_like(p)
    out[keep] = p[keep] / mass
    ent = float(-np.sum(out[keep] * np.log(out[keep])))

    return RichResult(
        title="Top-k sampling",
        summary_lines=[("k", k), ("Kept mass", mass), ("Entropy (nats)", ent)],
        payload={
            "probabilities": out.tolist(),
            "kept_indices": keep.astype(int).tolist(),
            "kept_mass": mass,
            "full_probabilities": p.tolist(),
            "entropy": ent,
            "estimate": out.tolist(),
            "n": int(z.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grtop: keep top k, renormalise by their mass, zero elsewhere; k=1 is greedy"
