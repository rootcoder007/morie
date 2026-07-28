# morie.fn -- function file (rootcoder007/morie)
"""EstimateScores: attention scores from QJL-sketched keys."""

import numpy as np

from ._richresult import RichResult

__all__ = ["turboquant_estimate_scores"]


def turboquant_estimate_scores(q, k_tildes, norms, S, scale=None,
                               softmax=True):
    r"""Reconstruct a row of attention scores from sketched keys.

    TurboQuant Algorithm 1, EstimateScores:

    .. math::
       \widehat{\mathrm{score}}_j = \frac{1}{m}\sqrt{\frac{\pi}{2}}\,
         \nu_j \, \langle S q_n,\ \tilde k_j\rangle,
       \qquad j = 1,\ldots,n

    with :math:`\tilde k_j` the sign bits of the sketched key and
    :math:`\nu_j` its stored norm. The projection :math:`Sq` is
    computed ONCE and reused across all :math:`n` cached keys, which is
    what makes the decode cost :math:`O(md + nm)` instead of
    :math:`O(nd)` -- the saving that motivates the whole scheme, since
    :math:`n` grows with context length while :math:`m` does not.

    The softmax is where the error budget actually lands. Absolute
    error in a score matters only relative to the score GAPS: an
    estimate that misranks two nearly-tied keys changes the output very
    little, while one that perturbs the top score shifts the whole
    distribution. ``max_abs_error_tolerable`` reports the smallest gap
    between the top scores, which is the scale the sketch error must
    stay below to preserve the ranking.

    Parameters
    ----------
    q : array-like, shape (d,)
    k_tildes : array-like of {-1, +1}, shape (n, m)
    norms : array-like, shape (n,)
    S : array-like, shape (m, d)
    scale : float, optional
        Attention temperature; :math:`1/\sqrt{d}` by default.
    softmax : bool
        Also return the normalised weights.

    Returns
    -------
    RichResult
        ``scores``, ``weights``, ``top_index``, ``entropy``,
        ``effective_context``, ``max_abs_error_tolerable``.

    References
    ----------
    Zandieh, Daliri and Han (2024), arXiv:2406.03482, Algorithm 1.
    Zandieh et al. (2026), TurboQuant, ICLR, arXiv:2504.19874.

    Examples
    --------
    >>> import numpy as np
    >>> S = np.eye(2)
    >>> out = turboquant_estimate_scores([1.0, 0.0], [[1, 1], [-1, 1]],
    ...                                  [1.0, 1.0], S)
    >>> int(out["top_index"])
    0
    """
    qv = np.asarray(q, dtype=float).ravel()
    Sm = np.atleast_2d(np.asarray(S, dtype=float))
    m, d = Sm.shape
    if qv.size != d:
        raise ValueError("q has dimension %d, S expects %d." % (qv.size, d))
    K = np.atleast_2d(np.asarray(k_tildes, dtype=float))
    if K.shape[1] != m:
        raise ValueError(
            "k_tildes has %d columns, S has %d rows." % (K.shape[1], m)
        )
    if not np.all(np.isin(K, (-1.0, 1.0))):
        raise ValueError("k_tildes must contain only -1 and +1.")
    nu = np.asarray(norms, dtype=float).ravel()
    if nu.size != K.shape[0]:
        raise ValueError(
            "norms has %d entries for %d keys." % (nu.size, K.shape[0])
        )
    sc = 1.0 / np.sqrt(d) if scale is None else float(scale)

    Sq = Sm @ qv                      # computed once, reused for every key
    raw = np.sqrt(np.pi / 2.0) / m * nu * (K @ Sq)
    scores = raw * sc

    payload = {
        "estimate": scores,
        "scores": scores,
        "raw_inner_products": raw,
        "projected_query": Sq,
        "top_index": int(np.argmax(scores)),
        "scale": sc,
        "cost_note": (
            "S q is formed once and reused, so the decode is O(md + nm) "
            "rather than O(nd); n grows with context length and m does not"
        ),
        "m": int(m),
        "d": int(d),
        "n_keys": int(K.shape[0]),
        "method": "TurboQuant EstimateScores (Algorithm 1)",
    }
    if scores.size > 1:
        srt = np.sort(scores)[::-1]
        payload["max_abs_error_tolerable"] = float(srt[0] - srt[1])
        payload["error_note"] = (
            "the gap between the top two scores: sketch error below this "
            "leaves the ranking intact, error above it can change which key "
            "dominates the softmax"
        )
    if softmax:
        mx = scores.max()
        ex = np.exp(scores - mx)
        w = ex / ex.sum()
        nz = w[w > 0]
        payload.update({
            "weights": w,
            "entropy": float(-np.sum(nz * np.log(nz))),
            "effective_context": float(np.exp(-np.sum(nz * np.log(nz)))),
            "effective_note": (
                "exponentiated entropy of the attention weights: how many "
                "keys the head is effectively reading"
            ),
        })
    return RichResult(payload=payload)


def cheatsheet():
    return (
        "tqest: attention scores from sketched keys, with the score gap that "
        "bounds tolerable sketch error"
    )
