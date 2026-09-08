# morie.fn -- function file (rootcoder007/morie)
"""Natural indirect effect (Pearl 2001).

DUPLICATE: the natural indirect effect is already implemented in
``tmlnie`` (public name ``nieff``).  Per ledger/wave2/DUPMAP.tsv this
module aliases that implementation rather than carrying a second copy.
"""

from .tmlnie import nieff as _nieff

__all__ = ["natural_indirect_effect"]


def natural_indirect_effect(y11, y10):
    """Natural indirect effect ``E[Y(1, M(1))] - E[Y(1, M(0))]``.

    Alias of :func:`morie.fn.tmlnie.nieff`.  The two counterfactual
    outcome vectors are paired per unit, so the contrast is taken
    within unit and its standard error is the paired one.

    Parameters
    ----------
    y11 : array-like
        Unit-level ``Y(1, M(1))``.
    y10 : array-like
        Unit-level ``Y(1, M(0))``, the cross-world outcome.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``mean_y11``, ``mean_y10``, ``n``.

    References
    ----------
    Pearl, J. (2001).  Direct and indirect effects.  Proceedings of the
    Seventeenth Conference on Uncertainty in Artificial Intelligence,
    411--420.  Morgan Kaufmann.
    """
    return _nieff(y11, y10)


def cheatsheet():
    return "nie: Natural indirect effect (alias of tmlnie.nieff)"


naturalindirecteffect = natural_indirect_effect
