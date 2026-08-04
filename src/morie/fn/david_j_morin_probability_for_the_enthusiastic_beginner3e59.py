"""Deprecated alias for :func:`morie.fn.pmfvar`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .pmfvar import pmfvar as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_59"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_59(values, probs):
    """Deprecated; use :func:`morie.fn.pmfvar` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_59() is the book-coordinate name for pmfvar(); "
        "it will be removed. Use morie.fn.pmfvar() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(values, probs)
