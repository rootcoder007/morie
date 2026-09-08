"""Deprecated alias for :func:`morie.fn.sd_of_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sd_of_mean import sd_of_mean as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_53"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_53(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.sd_of_mean` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_53() is the book-coordinate name for sd_of_mean(); "
        "it will be removed. Use morie.fn.sd_of_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
