"""Deprecated alias for :func:`morie.fn.slopeprod`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .slopeprod import slopeprod as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_53"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_53(x, y):
    """Deprecated; use :func:`morie.fn.slopeprod` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_53() is the book-coordinate name for slopeprod(); "
        "it will be removed. Use morie.fn.slopeprod() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(x, y)
