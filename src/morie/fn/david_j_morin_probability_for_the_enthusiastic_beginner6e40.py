"""Deprecated alias for :func:`morie.fn.regression_to_mean_factor`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .regression_to_mean_factor import regression_to_mean_factor as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_40"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_40(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.regression_to_mean_factor` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_40() is the book-coordinate name for regression_to_mean_factor(); "
        "it will be removed. Use morie.fn.regression_to_mean_factor() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
