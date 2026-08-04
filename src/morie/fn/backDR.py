# morie.fn -- function file (rootcoder007/morie)
"""Back-door adjustment -- front-end to bdrj."""

__all__ = ["back_door"]


def back_door(Y, X, C):
    r"""Back-door adjustment of the X -> Y effect for the discrete set C.

    Delegates to :func:`morie.fn.bdrj.backdoor_adjustment_formula`
    (Pearl 2009, Thm 3.3.2); whether C is a VALID adjustment set is a
    graph question answered by :func:`morie.fn.bdcrt.backdoor_criterion`,
    not by this arithmetic. The placeholder this replaces averaged Y.
    """
    from .bdrj import backdoor_adjustment_formula

    return backdoor_adjustment_formula(X, Y, C)


def cheatsheet():
    return "backDR: back-door adjustment (front-end to bdrj)"


# compact alias per ledger/NAMING.md
backdoor = back_door
