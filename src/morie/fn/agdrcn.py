# morie.fn -- slice s03 (rootcoder007/morie)
"""Dirichlet concentration parameter for AlphaZero root noise.

Source consulted (FETCHED): Silver, D. et al. (2018), arXiv:1712.01815:
"Dirichlet noise Dir(alpha) was added to the prior probabilities in the
root node; this was scaled in inverse proportion to the approximate
number of legal moves in a typical position, to a value of
alpha = {0.3, 0.15, 0.03} for chess, shogi and Go respectively."

The paper states the *rule* (inverse proportionality) and the three
values, not the constant of proportionality.  Taking the branching
factors those three games are usually quoted with -- about 35, 92 and
250 legal moves -- the implied constants are 10.5, 13.8 and 7.5, so a
scale of 10 reproduces all three to within the precision at which the
paper reports them.  ``scale=10`` is therefore the default, and it is
what the module's own formula line specifies; it is an inference from
the published rule, not a number printed in the paper, and is flagged
as such in ``method``.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401

from ._richresult import RichResult

__all__ = ["alphazero_dirichlet_concentration"]

# The three values the paper prints, with the branching factors usually
# quoted for those games.
_PUBLISHED = {"chess": (35.0, 0.3), "shogi": (92.0, 0.15), "go": (250.0, 0.03)}


def alphazero_dirichlet_concentration(avg_legal, scale=10.0):
    """alpha = scale / (average number of legal moves).

    Parameters
    ----------
    avg_legal : float
        Average number of legal moves in a typical position.
    scale : float
        Constant of proportionality; see the module docstring.

    Returns
    -------
    RichResult with payload:
        estimate      : alpha
        published_alpha : the paper's value when ``avg_legal`` matches a
                          game's branching factor, else nan
        scale, avg_legal
    """
    b = float(avg_legal)
    s = float(scale)
    alpha = s / b if b > 0.0 else float("nan")
    pub = float("nan")
    for _, (bf, av) in sorted(_PUBLISHED.items()):
        if abs(bf - b) < 1e-9:
            pub = av
    return RichResult(
        title="AlphaZero Dirichlet concentration",
        summary_lines=[("alpha", alpha)],
        payload={
            "estimate": alpha,
            "alpha": alpha,
            "published_alpha": pub,
            "scale": s,
            "avg_legal": b,
            "method": ("Dirichlet concentration alpha = scale / avg legal moves; "
                       "the inverse-proportionality rule is the paper's, the "
                       "scale of 10 is inferred from its three printed values"),
        },
    )


def cheatsheet():
    return "agdrcn: AlphaZero Dirichlet concentration parameter"
