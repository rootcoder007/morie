"""Deprecated alias for :func:`morie.fn.pascalid`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .pascalid import pascalid as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_60"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_60(n, k):
    """Deprecated; use :func:`morie.fn.pascalid` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_60() is the book-coordinate name for pascalid(); "
        "it will be removed. Use morie.fn.pascalid() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n, k)
