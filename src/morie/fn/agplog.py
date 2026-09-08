# morie.fn -- slice s03 (rootcoder007/morie)
"""Structured play log for one game.

There is no equation here either.  Silver et al. (2018),
arXiv:1712.01815 (FETCHED), reports games and their outcomes but
specifies no log format, so nothing is attributed to it.  What the
function provides is a canonical, order-stable record of a game --
move index, action, root visit count of the chosen move, and the value
estimate -- together with the summary statistics that a log is actually
read for: the game length, the total and mean value, and the entropy of
the realised move distribution.

The digest is the same Rabin-Karp polynomial hash as ``agdpck``,
h <- (131 h + c) mod (2^31 - 1), exact in double precision.  Writing to
disk is opt-in; by default nothing touches the filesystem.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_play_log"]

_MOD = 2147483647.0
_BASE = 131.0


def _digest(text):
    h = 0.0
    for ch in text:
        h = (_BASE * h + float(ord(ch))) % _MOD
    return h


def alphazero_play_log(game, path=None, values=None, visits=None):
    """Canonicalise a game record and summarise it.

    Parameters
    ----------
    game : array-like
        The actions played, in order.
    path : str, optional
        When given, the canonical text is written there.
    values : array-like, optional
        Value estimate at each move.
    visits : array-like, optional
        Root visit count of the played move, at each move.

    Returns
    -------
    RichResult with payload:
        estimate  : the digest
        moves     : number of moves
        mean_value, total_value
        action_entropy : entropy of the empirical action distribution
    """
    acts = k.vec(game)
    n = len(acts)
    v = k.vec(values) if values is not None else [0.0] * n
    vis = k.vec(visits) if visits is not None else [0.0] * n
    lines = []
    for i in range(n):
        lines.append("%d,%.17g,%.17g,%.17g" % (i, acts[i], vis[i], v[i]))
    text = "\n".join(lines)
    written = False
    if path is not None:
        with open(path, "w") as fh:
            fh.write(text)
        written = True
    counts = {}
    for a in acts:
        counts[a] = counts.get(a, 0.0) + 1.0
    h = 0.0
    for key in sorted(counts):
        q = counts[key] / n if n else 0.0
        if q > 0.0:
            h -= q * math.log(q)
    tot = 0.0
    for x in v:
        tot += x
    return RichResult(
        title="AlphaZero play log",
        summary_lines=[("moves", n)],
        payload={
            "estimate": _digest(text),
            "digest": _digest(text),
            "moves": n,
            "total_value": tot,
            "mean_value": tot / n if n else float("nan"),
            "action_entropy": h,
            "written": written,
            "method": "Canonical game log with a Rabin-Karp digest",
        },
    )


def cheatsheet():
    return "agplog: AlphaZero play log structured output"
