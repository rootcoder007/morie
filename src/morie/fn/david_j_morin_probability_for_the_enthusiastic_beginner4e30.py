"""Deprecated alias for :func:`morie.fn.exponential_crossing_time`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .exponential_crossing_time import exponential_crossing_time as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_30"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_30(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.exponential_crossing_time` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_30() is the book-coordinate name for exponential_crossing_time(); "
        "it will be removed. Use morie.fn.exponential_crossing_time() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
