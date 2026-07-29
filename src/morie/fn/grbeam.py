# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Beam search decoder: keep top-k hypotheses at each step."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_beam_search_decoder"]

_METHOD = "Beam search decoding"


def geron_beam_search_decoder(scores, beam_width, max_len=None, length_penalty=0.0):
    r"""Decode the highest-scoring sequence by beam search.

    .. math::
        B_t = \operatorname*{top-k}_{h \in B_{t-1},\, y}
              \bigl[\text{score}(h) + \log p(y \mid h)\bigr]

    Beam search is greedy decoding with :math:`k` regrets kept open.  It
    is *not* exact: the optimal sequence can fall off the beam at an
    early step where it looks bad, which is why widening the beam
    sometimes lowers quality rather than raising it.

    Parameters
    ----------
    scores : array-like, shape (T, V)
        Per-step log-probabilities over the vocabulary. Rows need not be
        normalised, but they must be log-scale (non-positive if they are
        proper log-probabilities).
    beam_width : int
        Number of hypotheses to keep, at least 1.
    max_len : int, optional
        Stop after this many steps; defaults to ``T``.
    length_penalty : float, optional
        Exponent :math:`\alpha` in the GNMT normaliser
        ``score / len**alpha`` used for the final ranking. Zero (the
        default) means raw log-probability.

    Returns
    -------
    RichResult
        Payload keys ``best_sequence``, ``best_score``,
        ``beams`` (list of ``(sequence, score)``, best first),
        ``normalised_scores``, ``greedy_sequence``, ``estimate``,
        ``n``, ``method``.

    References
    ----------
    Géron Ch 14, Beam Search section.

    Examples
    --------
    >>> r = geron_beam_search_decoder([[-0.1, -2.0], [-0.2, -3.0]], beam_width=2)
    >>> r["best_sequence"]
    [0, 0]
    >>> round(r["best_score"], 6)
    -0.3
    >>> [ (s, round(v, 6)) for s, v in r["beams"] ]
    [([0, 0], -0.3), ([1, 0], -2.2)]
    """
    S = np.atleast_2d(np.asarray(scores, dtype=float))
    if S.ndim != 2 or S.size == 0:
        raise ValueError(f"scores must be a non-empty 2-D (T, V) array, got shape {S.shape}.")
    if not np.all(np.isfinite(S)):
        raise ValueError("scores contains non-finite values; use a large negative number, not -inf.")
    T, V = S.shape
    beam_width = int(beam_width)
    if beam_width < 1:
        raise ValueError(f"beam_width must be at least 1, got {beam_width}.")
    if max_len is None:
        steps = T
    else:
        steps = int(max_len)
        if steps < 1:
            raise ValueError(f"max_len must be at least 1, got {max_len}.")
        if steps > T:
            raise ValueError(
                f"max_len={steps} exceeds the {T} steps of scores supplied."
            )
    length_penalty = float(length_penalty)
    if length_penalty < 0:
        raise ValueError(f"length_penalty must be non-negative, got {length_penalty}.")

    beams = [([], 0.0)]
    for t in range(steps):
        cand = []
        for seq, sc in beams:
            for y in range(V):
                cand.append((seq + [y], sc + float(S[t, y])))
        cand.sort(key=lambda p: -p[1])
        beams = cand[:beam_width]

    denom = float(steps) ** length_penalty if length_penalty else 1.0
    ranked = sorted(beams, key=lambda p: -(p[1] / denom))
    best_seq, best_score = ranked[0]
    greedy = [int(np.argmax(S[t])) for t in range(steps)]

    return RichResult(
        title="Beam search decoder",
        summary_lines=[("Beam width", beam_width), ("Best score", best_score)],
        payload={
            "best_sequence": [int(y) for y in best_seq],
            "best_score": float(best_score),
            "beams": [([int(y) for y in s], float(v)) for s, v in ranked],
            "normalised_scores": [float(v) / denom for _, v in ranked],
            "greedy_sequence": greedy,
            "beam_width": beam_width,
            "estimate": float(best_score),
            "n": steps,
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grbeam: beam search -- keep top-k hypotheses by cumulative log-prob at each step"
