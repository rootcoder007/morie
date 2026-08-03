"""Deprecated alias for :func:`morie.fn.conditional_from_joint`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .conditional_from_joint import conditional_from_joint as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_48"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_48(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.conditional_from_joint` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_48() is the book-coordinate name for conditional_from_joint(); "
        "it will be removed. Use morie.fn.conditional_from_joint() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
