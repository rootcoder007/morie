# morie.fn -- function file (rootcoder007/morie)
"""Rotary position embedding."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["rope", "rotary_position_embedding"]


def rope(q, m, theta):
    """Rotary position embedding.

    Rotary position embedding: f(q, m) = R_{theta, m} q.

    Su et al. (2021), RoFormer.  Coordinate pairs (q_{2i}, q_{2i+1}) are
    rotated by angle m * theta_i.  Because the rotation is orthogonal,
    the inner product of two rotated vectors depends only on their
    relative position m - n, which is the property the construction was
    built to get.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Rotary position embedding", payload=_c.rope(q=q, m=m, theta=theta))


rotary_position_embedding = rope


def cheatsheet():
    return "atrope: Rotary position embedding"
