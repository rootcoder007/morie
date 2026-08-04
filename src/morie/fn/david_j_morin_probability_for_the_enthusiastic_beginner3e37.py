"""Deprecated alias for :func:`morie.fn.popvar`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .popvar import popvar as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_37"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_37(x):
    """Deprecated; use :func:`morie.fn.popvar` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_37() is the book-coordinate name for popvar(); "
        "it will be removed. Use morie.fn.popvar() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x)
