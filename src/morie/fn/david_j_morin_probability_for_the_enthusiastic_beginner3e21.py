"""Deprecated alias for :func:`morie.fn.coinvar`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .coinvar import coinvar as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_21"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_21():
    """Deprecated; use :func:`morie.fn.coinvar` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_21() is the book-coordinate name for coinvar(); "
        "it will be removed. Use morie.fn.coinvar() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl()
