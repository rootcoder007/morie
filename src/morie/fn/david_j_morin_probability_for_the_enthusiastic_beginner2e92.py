"""Deprecated alias for :func:`morie.fn.inclusion_exclusion_3`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .inclusion_exclusion_3 import inclusion_exclusion_3 as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_92"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_92(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.inclusion_exclusion_3` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_92() is the book-coordinate name for inclusion_exclusion_3(); "
        "it will be removed. Use morie.fn.inclusion_exclusion_3() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
