# morie.fn -- function file (rootcoder007/morie)
"""Trigram maximum-likelihood probability (Burkov eq 2.4)."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["burkov_lm_ch2_trigram_count"]


def burkov_lm_ch2_trigram_count(t_i, t_im1, t_im2, counts=None,
                                bigram_counts=None, vocab_size=None,
                                smoothing=0.0):
    r"""Maximum-likelihood trigram probability from corpus counts.

    Burkov equation (2.4), p. 77:

    .. math::
       \Pr(t_i \mid t_{i-2}, t_{i-1}) =
         \frac{C(t_{i-2}, t_{i-1}, t_i)}{C(t_{i-2}, t_{i-1})}

    The estimator's defect is structural, not a matter of having too
    little data: any trigram absent from the corpus receives
    probability EXACTLY ZERO, so a single unseen triple makes the
    likelihood of a whole sentence zero and its perplexity infinite.
    Since the number of possible trigrams is :math:`V^3` and no corpus
    covers a meaningful fraction of it, this is the normal case rather
    than an edge case.

    ``smoothing`` applies add-:math:`\alpha` (Laplace when
    :math:`\alpha = 1`), which requires ``vocab_size`` because the
    denominator gains :math:`\alpha V`. It is the crudest fix and it
    is reported as such: it moves a great deal of mass onto unseen
    events, which is why Kneser-Ney and its relatives replaced it.

    Parameters
    ----------
    t_i, t_im1, t_im2 : hashable or sequence
        The target token and its two predecessors.
    counts : mapping, optional
        Trigram counts keyed by ``(t_im2, t_im1, t_i)``.
    bigram_counts : mapping, optional
        Context counts keyed by ``(t_im2, t_im1)``. Derived from
        ``counts`` when omitted.
    vocab_size : int, optional
        Required when ``smoothing`` is non-zero.
    smoothing : float
        Add-alpha constant.

    Returns
    -------
    RichResult
        ``probability``, ``trigram_count``, ``context_count``,
        ``unseen``, ``smoothed``.

    References
    ----------
    Burkov (2025), *The Hundred-Page Language Models Book*, chapter 2,
    equation (2.4), p. 77.

    Examples
    --------
    >>> c = {("a", "b", "c"): 3, ("a", "b", "d"): 1}
    >>> out = burkov_lm_ch2_trigram_count("c", "b", "a", counts=c)
    >>> float(out["probability"])
    0.75
    """
    if counts is None:
        raise ValueError(
            "counts is required: a trigram probability cannot be formed "
            "without corpus counts."
        )
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")
    if smoothing > 0 and vocab_size is None:
        raise ValueError(
            "vocab_size is required when smoothing is non-zero, because the "
            "denominator gains alpha * V."
        )
    key = (t_im2, t_im1, t_i)
    ctx = (t_im2, t_im1)
    tri = float(counts.get(key, 0))
    if bigram_counts is not None:
        den = float(bigram_counts.get(ctx, 0))
    else:
        den = float(sum(v for k, v in counts.items()
                        if len(k) == 3 and (k[0], k[1]) == ctx))
    unseen = tri == 0
    if smoothing > 0:
        V = int(vocab_size)
        prob = (tri + smoothing) / (den + smoothing * V)
    else:
        prob = tri / den if den > 0 else np.nan
    return RichResult(
        payload={
            "estimate": float(prob),
            "probability": float(prob),
            "trigram_count": tri,
            "context_count": den,
            "unseen": bool(unseen),
            "context_unseen": bool(den == 0),
            "smoothed": bool(smoothing > 0),
            "smoothing": float(smoothing),
            "zero_note": (
                "an unsmoothed MLE gives probability exactly zero to any "
                "unseen trigram, so one unseen triple makes a sentence's "
                "likelihood zero and its perplexity infinite; with V^3 "
                "possible trigrams this is the normal case"
            ),
            "smoothing_note": (
                None if smoothing == 0 else
                "add-alpha is the crudest remedy and moves a great deal of "
                "mass onto unseen events; Kneser-Ney and its relatives exist "
                "because of that"
            ),
            "method": "Trigram MLE probability (Burkov eq 2.4)",
        }
    )


def cheatsheet():
    return "b204: trigram MLE with the zero-probability problem made explicit"
