# morie.fn -- function file (rootcoder007/morie)
"""Front-door adjustment -- front-end to fdadj."""

__all__ = ["front_door"]


def front_door(Y, X, M):
    r"""Front-door adjustment of the X -> Y effect through mediator M.

    Delegates to :func:`morie.fn.fdadj.frontdoor_adjustment` (Pearl
    2009, Thm. 3.3.4). The placeholder this replaces averaged Y.
    """
    from .fdadj import frontdoor_adjustment

    return frontdoor_adjustment(X, M, Y)


def cheatsheet():
    return "medFront: front-door adjustment (front-end to fdadj)"
