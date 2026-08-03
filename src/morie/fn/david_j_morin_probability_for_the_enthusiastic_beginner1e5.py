"""Deprecated alias for :func:`morie.fn.partial_permutations`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .partial_permutations import partial_permutations as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_5"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_5(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.partial_permutations` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_5() is the book-coordinate name for partial_permutations(); "
        "it will be removed. Use morie.fn.partial_permutations() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
