"""Deprecated alias for :func:`morie.fn.sdfromvar`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sdfromvar import sdfromvar as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_39"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_39(var_x):
    """Deprecated; use :func:`morie.fn.sdfromvar` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_39() is the book-coordinate name for sdfromvar(); "
        "it will be removed. Use morie.fn.sdfromvar() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(var_x)
