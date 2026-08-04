"""Deprecated alias for :func:`morie.fn.binommom2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binommom2 import binommom2 as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_66"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_66(n, p):
    """Deprecated; use :func:`morie.fn.binommom2` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_66() is the book-coordinate name for binommom2(); "
        "it will be removed. Use morie.fn.binommom2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n, p)
