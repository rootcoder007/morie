# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Beam search decoding with beam width K."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_beam_search"]


def geron_beam_search(model, src, beam_width=3, max_len=10, eos=None, length_penalty=0.0):
    """
    Beam search decoding with beam width K.

    Formula: maintain top-K partial hypotheses by score

    `model` is ``model(src, prefix) -> log_probs`` returning a 1-D array of
    log-probabilities over the vocabulary for the next token given the
    prefix (a tuple of ints). The scorer is checked for finiteness and for
    normalisation, because an unnormalised scorer silently turns beam search
    into an arbitrary greedy walk.

    Parameters
    ----------
    model : callable
        Next-token log-probability scorer as described above.
    src : any
        Source input, passed through to `model` unchanged.
    beam_width : int
        Number of hypotheses kept per step (>= 1). At beam_width equal to
        the vocabulary size the search is exhaustive.
    max_len : int
        Maximum number of tokens to generate (>= 1).
    eos : int, optional
        End-of-sequence token; hypotheses that emit it are finished.
    length_penalty : float
        Exponent alpha in the score normaliser ``score / len**alpha``
        used for the final ranking (0 disables it).

    Returns
    -------
    result : RichResult
        Keys: sequence, score, beams, scores, finished, estimate, n, method.

    Examples
    --------
    A prefix-independent scorer with p = (0.6, 0.4): the best two-token
    sequence is (0, 0) with log score log(0.36):

    >>> lp = np.log([0.6, 0.4])
    >>> r = geron_beam_search(lambda s, prefix: lp, None, beam_width=2, max_len=2)
    >>> [int(t) for t in r["sequence"]]
    [0, 0]
    >>> round(float(r["score"]), 6)
    -1.021651
    >>> len(r["beams"])
    2

    With eos=1 the hypothesis can stop early, and stopping is preferred once
    continuing costs more than it gains:

    >>> r2 = geron_beam_search(lambda s, prefix: np.log([0.1, 0.9]), None, beam_width=2, max_len=3, eos=1)
    >>> [int(t) for t in r2["sequence"]]
    [1]

    References
    ----------
    Géron Ch 14
    """
    if not callable(model):
        raise ValueError("geron_beam_search: model must be callable")
    K = int(beam_width)
    if K < 1:
        raise ValueError("geron_beam_search: beam_width must be >= 1")
    L = int(max_len)
    if L < 1:
        raise ValueError("geron_beam_search: max_len must be >= 1")

    def score_next(prefix):
        lp = np.asarray(model(src, tuple(prefix)), dtype=float).ravel()
        if lp.size == 0:
            raise ValueError("geron_beam_search: model returned an empty log-probability vector")
        if not np.all(np.isfinite(lp[lp > -np.inf])):
            raise ValueError("geron_beam_search: model returned non-finite log-probabilities")
        total = float(np.sum(np.exp(lp)))
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"geron_beam_search: model log-probabilities exponentiate to {total!r}, not 1; "
                "beam search requires a normalised next-token distribution"
            )
        return lp

    live = [((), 0.0)]
    finished = []
    for _ in range(L):
        cand = []
        for prefix, sc in live:
            lp = score_next(prefix)
            V = lp.size
            if eos is not None and not (0 <= int(eos) < V):
                raise ValueError(f"geron_beam_search: eos={eos} is outside the vocabulary of size {V}")
            top = np.argsort(-lp, kind="mergesort")[: min(K, V)]
            for tok in top:
                cand.append((prefix + (int(tok),), sc + float(lp[tok])))
        if not cand:
            break
        cand.sort(key=lambda ps: -ps[1])
        live = []
        for seq, sc in cand:
            if eos is not None and seq[-1] == int(eos):
                finished.append((seq, sc))
            else:
                live.append((seq, sc))
            if len(live) >= K:
                break
        if not live:
            break
    finished.extend(live)

    def normed(item):
        seq, sc = item
        return sc / (len(seq) ** length_penalty) if length_penalty else sc

    finished.sort(key=lambda it: -normed(it))
    best_seq, best_score = finished[0]

    return RichResult(
        title="Beam search decoding",
        summary_lines=[("Beam width", K), ("Best length", len(best_seq)), ("Log score", best_score)],
        payload={
            "sequence": list(best_seq),
            "score": float(best_score),
            "beams": [list(s) for s, _ in finished[:K]],
            "scores": [float(sc) for _, sc in finished[:K]],
            "finished": len(finished),
            "estimate": float(best_score),
            "n": int(len(best_seq)),
            "method": f"Beam search with width {K}",
        },
    )


def cheatsheet():
    return "hmbms: Beam search decoding with beam width K"
