# morie.fn -- function file (rootcoder007/morie)
"""Vote trading (logrolling) model"""

from __future__ import annotations

from ._containers import DescriptiveResult
from .spatialvote import vote_trading_riker_brams


def vote_trading(valuations):
    r"""Vote trading (logrolling) model: Riker-Brams vote trading over binary issues, number of trades as ``value``.

    ``valuations[i][k]`` is voter ``i``'s signed intensity for passing issue
    ``k``. Delegates to :func:`morie.fn.spatialvote.vote_trading_riker_brams`;
    ``extra`` carries sincere and post-trade outcomes, the trades and total
    welfare before and after (the paradox of vote trading: it can fall).

    References
    ----------
    Riker, W. H. and Brams, S. J. (1973). The paradox of vote trading.
    *American Political Science Review*, 67(4), 1235-1247.

    Examples
    --------
    >>> r = vote_trading([[3, -1], [-1, 3], [-3, -3]])
    >>> r.value, r.extra["welfare_before"], r.extra["welfare_after"]
    (1, 0.0, -2.0)
    """
    V = valuations.tolist() if hasattr(valuations, "tolist") else [list(r) for r in valuations]
    r = vote_trading_riker_brams(V)
    return DescriptiveResult(name="svvtr", value=len(r["trades"]), extra=dict(r))


vote = vote_trading


def cheatsheet() -> str:
    return "vote_trading(valuations) -> Riker-Brams vote trading"


# compact alias per ledger/NAMING.md
votetrading = vote_trading
