# morie.fn -- slice s03 (rootcoder007/morie)
"""Elo rating from a match pool.

Source consulted (FETCHED): Silver, D. et al. (2018), arXiv:1712.01815,
which states its own convention verbatim: "We estimate the probability
that player a will defeat player b by a logistic function

    p(a defeats b) = 1 / (1 + exp(c_elo (e(b) - e(a))))

and estimate the ratings e(.) by Bayesian logistic regression, computed
by the BayesElo program using the standard constant c_elo = 1/400."

NOTE, and it matters: that is the *natural exponential* logistic, not
the classical Elo curve of Elo, A. (1978), *The Rating of Chessplayers,
Past and Present*, Arco, which is 1 / (1 + 10^((R_b - R_a)/400)).  The
two disagree -- a 200-point gap gives 0.622 under the paper's constant
but 0.760 under Elo's.  Both are provided; ``base="e"`` is AlphaZero's
own convention and the default, ``base=10`` is Elo's, and the docstring
does not pretend they are the same thing.

Given a score against an anchor of known rating the relation inverts in
closed form, which is what is done here; ratings are not fitted by
Bayesian logistic regression, and ``method`` says so.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_benchmark_eval"]


def alphazero_benchmark_eval(games, ladder=None, anchor=0.0, base="e",
                             c_elo=1.0 / 400.0):
    """Rating implied by a score against an anchor, and the ladder's expectations.

    Parameters
    ----------
    games : array-like or float
        Either the score in [0, 1], or a triple (wins, draws, losses).
    ladder : array-like, optional
        Ratings of other players; the expected score against each is
        returned.
    anchor : float
        Rating of the opponent the score was achieved against.
    base : {"e", 10}
        The logistic convention; see the module docstring.
    c_elo : float
        The constant; 1/400 in both conventions.

    Returns
    -------
    RichResult with payload:
        estimate  : the implied rating
        score     : the score used
        expected  : expected score against each ladder entry
        base
    """
    g = k.vec(games)
    if len(g) >= 3:
        w, d, l = g[0], g[1], g[2]
        tot = w + d + l
        score = (w + 0.5 * d) / tot if tot > 0.0 else float("nan")
    else:
        score = g[0] if g else float("nan")
    if score <= 0.0 or score >= 1.0:
        rating = float("-inf") if score <= 0.0 else float("inf")
    else:
        odds = math.log(score / (1.0 - score))
        rating = float(anchor) + (odds / c_elo if base == "e"
                                  else odds / (math.log(10.0) * c_elo))
    exp = []
    for r in (k.vec(ladder) if ladder is not None else []):
        d = c_elo * (r - rating)
        exp.append(1.0 / (1.0 + (math.exp(d) if base == "e" else 10.0 ** d)))
    return RichResult(
        title="Elo rating from a match pool",
        summary_lines=[("rating", rating), ("score", score)],
        payload={
            "estimate": rating,
            "rating": rating,
            "score": score,
            "expected": exp,
            "base": base,
            "method": ("Elo rating inverted in closed form from a score against an "
                       "anchor; AlphaZero's exp convention by default, Elo's "
                       "base-10 curve with base=10"),
        },
    )


def cheatsheet():
    return "agbnch: AlphaZero benchmark eval (ELO + Tactic suite)"
