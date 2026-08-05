"""Deprecated alias for :func:`morie.fn.retestiq`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .retestiq import retestiq as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_38"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_38():
    """Deprecated; use :func:`morie.fn.retestiq` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_38() is the book-coordinate name for retestiq(); "
        "it will be removed. Use morie.fn.retestiq() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl()
