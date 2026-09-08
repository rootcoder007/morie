# morie.fn -- slice s03 (rootcoder007/morie)
"""The AlphaZero policy-and-value loss.

Source consulted (FETCHED): Silver, D. et al. (2018), arXiv:1712.01815,
which prints the loss verbatim:

    l = (z - v)^2 - pi^T log p + c ||theta||^2

a squared error on the value head, a cross-entropy between the search
policy pi and the network policy p, and an L2 penalty on the
parameters.  Silver et al. (2017), *Nature* 550, 354-359, give the same
expression.  AlphaGo Zero used c = 1e-4.

The cross-entropy term is computed with the convention 0 log 0 = 0, so
that actions with zero search visits contribute nothing rather than a
NaN, and p is floored at a tiny epsilon to keep log p finite when the
policy head has collapsed.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["alphazero_value_head"]

_EPS = 1e-300


def alphazero_value_head(z, v, pi, p, theta=None, c=1e-4):
    """Total loss, and its three parts, for one training example.

    Parameters
    ----------
    z : float
        Game outcome in [-1, 1].
    v : float
        Value-head output.
    pi : array-like
        Search policy (normalised root visit counts).
    p : array-like
        Policy-head output.
    theta : array-like, optional
        Flattened parameters, for the L2 term.
    c : float
        L2 coefficient; AlphaGo Zero used 1e-4.

    Returns
    -------
    RichResult with payload:
        estimate  : the total loss l
        value_loss, policy_loss, l2 : its three parts
        sq_norm   : ||theta||^2
    """
    zz = float(z)
    vv = float(v)
    pp = k.vec(pi)
    qq = k.vec(p)
    vloss = (zz - vv) ** 2
    ploss = 0.0
    for i in range(len(pp)):
        if pp[i] > 0.0:
            ploss -= pp[i] * math.log(qq[i] if qq[i] > _EPS else _EPS)
    sq = 0.0
    if theta is not None:
        for x in k.vec(theta):
            sq += x * x
    l2 = float(c) * sq
    return RichResult(
        title="AlphaZero loss",
        summary_lines=[("total", vloss + ploss + l2)],
        payload={
            "estimate": vloss + ploss + l2,
            "value_loss": vloss,
            "policy_loss": ploss,
            "l2": l2,
            "sq_norm": sq,
            "method": "AlphaZero loss (z - v)^2 - pi' log p + c ||theta||^2",
        },
    )


def cheatsheet():
    return "agnnvh: AlphaZero policy + value head loss"
