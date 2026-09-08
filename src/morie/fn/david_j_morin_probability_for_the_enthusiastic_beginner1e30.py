"""Deprecated alias for :func:`morie.fn.tripnorep`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .tripnorep import tripnorep as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_30"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_30(N):
    """Deprecated; use :func:`morie.fn.tripnorep` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_30() is the book-coordinate name for tripnorep(); "
        "it will be removed. Use morie.fn.tripnorep() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(N)
