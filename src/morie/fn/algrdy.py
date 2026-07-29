# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Greedy decoding step (Alammar Ch 6)."""

import numpy as np

from ._richresult import RichResult

__all__ = ["alammar_greedy_decoding"]


def alammar_greedy_decoding(logits):
    """y_t = argmax over the vocabulary, per step.

    Accepts one step (a vector) or a sequence (a matrix, one row per
    step). Ties break to the LOWEST index, deterministically, and the
    payload says when a tie was broken -- silent tie-breaking is a
    reproducibility leak.

    Examples
    --------
    >>> alammar_greedy_decoding([[0.1, 2.0, 0.3]])["tokens"]
    [1]
    """
    Z = np.atleast_2d(np.asarray(logits, dtype=float))
    toks = []
    ties = []
    for row in Z:
        m = row.max()
        winners = np.flatnonzero(row == m)
        toks.append(int(winners[0]))
        ties.append(len(winners) > 1)
    return RichResult(payload={
        "tokens": toks, "had_ties": ties,
        "estimate": float(toks[0]), "n": len(toks),
        "method": "Greedy decoding argmax (Alammar Ch 6)"})


def cheatsheet():
    return "algrdy: per-step argmax, ties reported and broken low"
