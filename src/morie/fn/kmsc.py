# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-consistency decoding: majority vote over sampled reasoning
traces."""

from collections import Counter

from ._richresult import RichResult

__all__ = ["kamath_self_consistency"]


def kamath_self_consistency(samples, parse=None):
    """y_hat = mode over i of parse(sample_i).

    The reasoning traces are thrown away and only the ANSWERS are
    voted on -- that is the whole method. ``parse`` extracts the
    answer from a trace (identity by default). A trace whose parse
    returns None is counted as unparseable and excluded from the vote
    rather than voting for "None"; a tie is reported explicitly with
    the tied answers, and broken by first appearance, because a silent
    tie-break is a silent wrong answer.

    Reference: Kamath, Keenan, Somers and Sorenson (2024), *Large
    Language Models: A Deep Dive*, Springer, Ch 4, self-consistency
    (Wang et al. 2023).

    Examples
    --------
    >>> out = kamath_self_consistency(["A", "B", "A"])
    >>> out["answer"], out["votes"], out["tie"]
    ('A', 2, False)
    >>> abs(out["agreement"] - 2 / 3) < 1e-12
    True
    >>> tied = kamath_self_consistency(["B", "A"])
    >>> tied["answer"], tied["tie"], tied["tied_answers"]
    ('B', True, ['A', 'B'])
    >>> p = kamath_self_consistency(["ans: 4", "ans: 4", "ans: 5"],
    ...                             parse=lambda s: s.split()[-1])
    >>> p["answer"]
    '4'
    """
    traces = list(samples)
    if not traces:
        raise ValueError(
            "no samples; a majority vote over nothing has no winner.")
    if parse is not None and not callable(parse):
        raise ValueError("parse must be callable(sample) -> answer.")
    answers, unparsed = [], 0
    for s in traces:
        a = parse(s) if parse is not None else s
        if a is None:
            unparsed += 1
            continue
        answers.append(a)
    if not answers:
        raise ValueError(
            f"all {len(traces)} samples failed to parse; there is no "
            "answer to vote on.")
    counts = Counter(answers)
    top = max(counts.values())
    tied = [a for a in counts if counts[a] == top]
    # First appearance among the tied answers.
    winner = next(a for a in answers if a in tied)
    return RichResult(payload={
        "answer": winner, "votes": top,
        "agreement": top / len(answers),
        "counts": dict(counts),
        "tie": len(tied) > 1,
        "tied_answers": sorted(tied, key=repr),
        "n_unparsed": unparsed,
        "n_voted": len(answers),
        "estimate": top / len(answers),
        "n": len(traces),
        "method": "Self-consistency majority vote over parsed answers"})


def cheatsheet():
    return "kmsc: mode of parsed answers; ties and unparsed traces reported"
