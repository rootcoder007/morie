"""Deprecated alias for :func:`morie.fn.gausscount`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gausscount import gausscount as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_28"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_28(x, n_reps=100000, mu=35.0, sigma=5.4):
    """Deprecated; use :func:`morie.fn.gausscount` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_28() is the book-coordinate name for gausscount(); "
        "it will be removed. Use morie.fn.gausscount() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, n_reps, mu, sigma)
