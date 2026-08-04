"""Deprecated alias for :func:`morie.fn.piidint`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .piidint import piidint as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_95"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_95(p, k=2):
    """Deprecated; use :func:`morie.fn.piidint` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_95() is the book-coordinate name for piidint(); "
        "it will be removed. Use morie.fn.piidint() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(p, k)
