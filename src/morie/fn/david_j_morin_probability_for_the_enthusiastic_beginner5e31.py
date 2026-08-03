"""Deprecated alias for :func:`morie.fn.pmf_sd`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .pmf_sd import pmf_sd as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_31"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_31(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.pmf_sd` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_31() is the book-coordinate name for pmf_sd(); "
        "it will be removed. Use morie.fn.pmf_sd() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
