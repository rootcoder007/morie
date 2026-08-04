"""Deprecated alias for :func:`morie.fn.esumiid`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .esumiid import esumiid as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_15"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_15(e_x, n):
    """Deprecated; use :func:`morie.fn.esumiid` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_15() is the book-coordinate name for esumiid(); "
        "it will be removed. Use morie.fn.esumiid() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(e_x, n)
