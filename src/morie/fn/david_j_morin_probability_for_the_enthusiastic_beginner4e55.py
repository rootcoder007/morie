"""Deprecated alias for :func:`morie.fn.density_expectation`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .density_expectation import density_expectation as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_55"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_55(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.density_expectation` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_55() is the book-coordinate name for density_expectation(); "
        "it will be removed. Use morie.fn.density_expectation() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
