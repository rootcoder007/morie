# morie.fn -- function file (rootcoder007/morie)
"""Baron-Kenny four-step mediation -- front-end over bkmed."""

from __future__ import annotations

from .bkmed import baron_kenny as _baron_kenny

__all__ = ["baron_kenny_four_step"]


def baron_kenny_four_step(X, M, Y, alpha=0.05):
    """Baron & Kenny's four steps, treatment-first argument order.

    The same procedure as :func:`morie.fn.bkmed.baron_kenny`, which holds
    the implementation and already evaluates and reports all four
    conditions separately. This entry point exists for the
    treatment-first calling convention and does not carry a second copy
    of the logic.

    Returns the same result, with ``steps`` naming each of the four
    conditions. See :func:`morie.fn.bkmed.baron_kenny` for why step 1 --
    requiring a significant total effect -- is reported rather than
    enforced.
    """
    return _baron_kenny(Y, X, M, alpha=alpha)


def cheatsheet():
    return "bkfour: Baron-Kenny four-step mediation (treatment-first); see bkmed"
