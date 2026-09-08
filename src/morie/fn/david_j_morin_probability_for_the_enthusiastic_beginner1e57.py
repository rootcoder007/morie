"""Deprecated alias for :func:`morie.fn.stars_and_bars`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .stars_and_bars import stars_and_bars as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_57"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_57(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.stars_and_bars` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_57() is the book-coordinate name for stars_and_bars(); "
        "it will be removed. Use morie.fn.stars_and_bars() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
