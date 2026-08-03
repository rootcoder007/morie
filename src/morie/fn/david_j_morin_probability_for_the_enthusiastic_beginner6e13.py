"""Deprecated alias for :func:`morie.fn.slope_from_cov`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .slope_from_cov import slope_from_cov as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_13"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_13(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.slope_from_cov` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_13() is the book-coordinate name for slope_from_cov(); "
        "it will be removed. Use morie.fn.slope_from_cov() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
