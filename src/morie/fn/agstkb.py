# morie.fn -- slice s03 (rootcoder007/morie)
"""Head-to-head tally against a rating ladder.

Source consulted (FETCHED): Silver, D. et al. (2018), arXiv:1712.01815:
"Elo ratings were computed from the results of a 1 second per move
tournament between iterations of AlphaZero during training, and also a
baseline player: either Stockfish, Elmo or AlphaGo Lee respectively.
The Elo rating of the baseline players was anchored to publicly
available values."

So each baseline's rating is fixed and known, and the candidate's rating
follows from its score against each.  This function does that anchoring:
for every rung of the ladder it converts the (win, draw, loss) tally to
a score, inverts the logistic to an implied rating, and combines the
rungs by weighting each by the number of games played against it.  The
logistic convention is the paper's own p = 1/(1 + exp(c_elo (e(b) -
e(a)))) with c_elo = 1/400; see ``agbnch`` for why that is not the
classical base-10 Elo curve.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_stockfish_baseline"]


def alphazero_stockfish_baseline(games, ladder, base="e", c_elo=1.0 / 400.0):
    """Combine per-baseline tallies into one anchored rating.

    Parameters
    ----------
    games : 2-D array-like
        One row per rung: (wins, draws, losses) against that baseline.
    ladder : array-like
        Anchored rating of each rung, same length as ``games``.
    base : {"e", 10}
        Logistic convention.
    c_elo : float
        The constant; 1/400.

    Returns
    -------
    RichResult with payload:
        estimate   : games-weighted implied rating
        per_rung   : implied rating from each rung
        scores     : score against each rung
        n_games    : games against each rung
    """
    rows = k.mat(games)
    anchors = k.vec(ladder)
    per = []
    scores = []
    ns = []
    num = 0.0
    den = 0.0
    for i in range(len(rows)):
        w, d, l = rows[i][0], rows[i][1], rows[i][2]
        n = w + d + l
        ns.append(n)
        s = (w + 0.5 * d) / n if n > 0.0 else float("nan")
        scores.append(s)
        if 0.0 < s < 1.0:
            odds = math.log(s / (1.0 - s))
            r = anchors[i] + (odds / c_elo if base == "e"
                              else odds / (math.log(10.0) * c_elo))
        else:
            r = float("-inf") if s <= 0.0 else float("inf")
        per.append(r)
        if r == r and abs(r) != float("inf"):
            num += n * r
            den += n
    est = num / den if den > 0.0 else float("nan")
    wins = 0.0
    draws = 0.0
    losses = 0.0
    for row in rows:
        wins += row[0]
        draws += row[1]
        losses += row[2]
    return RichResult(
        title="Head-to-head against a rating ladder",
        summary_lines=[("rating", est)],
        payload={
            "estimate": est,
            "rating": est,
            "per_rung": per,
            "scores": scores,
            "n_games": ns,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "method": "Elo anchored to a ladder of baselines, games-weighted",
        },
    )


def cheatsheet():
    return "agstkb: AlphaZero vs Stockfish chess head-to-head"
