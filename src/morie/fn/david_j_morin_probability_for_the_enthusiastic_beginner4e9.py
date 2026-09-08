"""Deprecated alias for :func:`morie.fn.binompeq`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binompeq import binompeq as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9(n):
    """Deprecated; use :func:`morie.fn.binompeq` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_9() is the book-coordinate name for binompeq(); "
        "it will be removed. Use morie.fn.binompeq() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n)
