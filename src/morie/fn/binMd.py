# morie.fn -- function file (rootcoder007/morie)
"""Mediation for a binary outcome (logit) -- front-end over binmed."""

from __future__ import annotations

from .binmed import binary_outcome_mediation as _binmed

__all__ = ["binary_outcome_mediation"]


def binary_outcome_mediation(Y, X, M, C=None, **kwargs):
    """Binary-outcome mediation, outcome-first argument order.

    Identical estimator to :func:`morie.fn.binmed.binary_outcome_mediation`,
    which holds the implementation; only the argument order differs, this
    one leading with the outcome. It is kept as a separate entry point
    because callers written against either convention exist, and silently
    accepting a swapped order would be worse than having two names.

    See :func:`morie.fn.binmed.binary_outcome_mediation` for the method,
    the assumptions and the reference.
    """
    return _binmed(X, M, Y, C=C, **kwargs)


def cheatsheet():
    return "binMd: binary-outcome mediation (outcome-first order); see binmed"
