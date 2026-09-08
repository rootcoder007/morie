"""Deprecated alias for :func:`morie.fn.classify_events`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .classify_events import classify_events as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_24"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_24(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.classify_events` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_24() is the book-coordinate name for classify_events(); "
        "it will be removed. Use morie.fn.classify_events() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
