"""Deprecated alias for :func:`morie.fn.permutations_count`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .permutations_count import permutations_count as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_3"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.permutations_count` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_3() is the book-coordinate name for permutations_count(); "
        "it will be removed. Use morie.fn.permutations_count() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
