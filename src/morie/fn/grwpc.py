# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""WordPiece likelihood-maximizing merge score."""

import numpy as np

from ._richresult import RichResult

__all__ = ["geron_wordpiece_tokenizer_score"]

_METHOD = "WordPiece merge score"


def geron_wordpiece_tokenizer_score(counts, pairs):
    r"""Score adjacent symbol pairs the WordPiece way.

    .. math::
        \mathrm{score}(A, B) = \frac{\mathrm{count}(AB)}
                                    {\mathrm{count}(A)\,\mathrm{count}(B)}

    BPE merges the most *frequent* pair; WordPiece merges the pair whose
    joint count most exceeds what independence would predict.  The
    difference shows up on common function words: "th" and "e" are both
    everywhere, so ``the`` scores poorly despite being frequent, while a
    rare-but-inseparable pair scores highly.  That ratio is a
    likelihood-gain proxy, which is where the name
    "likelihood-maximizing" comes from.

    Parameters
    ----------
    counts : mapping
        Symbol -> count, positive.
    pairs : mapping or sequence
        ``(A, B) -> count(AB)``, or a sequence of ``(A, B, count)``.

    Returns
    -------
    RichResult
        Payload keys ``scores`` (pair -> score), ``best_pair``,
        ``best_score``, ``ranking``, ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 14, WordPiece section.

    Examples
    --------
    ``("h", "u")`` appears 10 times out of 15 h's and 20 u's, so it
    scores ``10/300``; the rarer but tighter ``("g", "s")`` beats it:

    >>> counts = {"h": 15, "u": 20, "g": 4, "s": 5}
    >>> pairs = {("h", "u"): 10, ("g", "s"): 4}
    >>> r = geron_wordpiece_tokenizer_score(counts, pairs)
    >>> round(r["scores"][("h", "u")], 6)
    0.033333
    >>> r["best_pair"]
    ('g', 's')
    >>> round(r["best_score"], 6)
    0.2

    A pair whose parts are missing from ``counts`` is an error, not a
    zero:

    >>> geron_wordpiece_tokenizer_score({"h": 1}, {("h", "z"): 1})
    Traceback (most recent call last):
        ...
    ValueError: symbol 'z' from pair ('h', 'z') is absent from counts.
    """
    if not isinstance(counts, dict):
        try:
            counts = dict(counts)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("counts must be a mapping from symbol to count.") from exc
    if not counts:
        raise ValueError("counts is empty.")
    for k, v in counts.items():
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"count for {k!r} must be positive, got {v}.")

    if isinstance(pairs, dict):
        items = list(pairs.items())
    else:
        items = []
        for row in pairs:
            if len(row) != 3:
                raise ValueError(
                    "sequence form of pairs must hold (A, B, count) triples."
                )
            items.append(((row[0], row[1]), row[2]))
    if not items:
        raise ValueError("pairs is empty; nothing to score.")

    scores = {}
    for (a, b), c in items:
        for sym in (a, b):
            if sym not in counts:
                raise ValueError(f"symbol {sym!r} from pair {(a, b)!r} is absent from counts.")
        c = float(c)
        if not np.isfinite(c) or c <= 0:
            raise ValueError(f"pair count for {(a, b)!r} must be positive, got {c}.")
        if c > min(counts[a], counts[b]):
            raise ValueError(
                f"pair {(a, b)!r} occurs {c:g} times but {a!r}/{b!r} occur "
                f"{counts[a]:g}/{counts[b]:g}; a pair cannot outnumber its parts."
            )
        scores[(a, b)] = c / (float(counts[a]) * float(counts[b]))

    ranking = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    best_pair, best_score = ranking[0]

    return RichResult(
        title="WordPiece merge score",
        summary_lines=[("Best pair", str(best_pair)), ("Score", float(best_score))],
        payload={
            "scores": scores,
            "best_pair": best_pair,
            "best_score": float(best_score),
            "ranking": [(p, float(s)) for p, s in ranking],
            "estimate": float(best_score),
            "n": len(scores),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grwpc: score(A,B) = count(AB)/(count(A)count(B)); BPE takes frequency, WordPiece takes surprise"
