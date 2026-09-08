"""Deprecated alias for :func:`morie.fn.twocoinvar`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .twocoinvar import twocoinvar as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_28"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_28():
    """Deprecated; use :func:`morie.fn.twocoinvar` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_28() is the book-coordinate name for twocoinvar(); "
        "it will be removed. Use morie.fn.twocoinvar() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl()
