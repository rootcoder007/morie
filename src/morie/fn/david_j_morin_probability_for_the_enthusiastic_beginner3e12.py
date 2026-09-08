"""Deprecated alias for :func:`morie.fn.esumconv`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .esumconv import esumconv as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_12"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_12(values_x, probs_x, values_y, probs_y):
    """Deprecated; use :func:`morie.fn.esumconv` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_12() is the book-coordinate name for esumconv(); "
        "it will be removed. Use morie.fn.esumconv() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(values_x, probs_x, values_y, probs_y)
