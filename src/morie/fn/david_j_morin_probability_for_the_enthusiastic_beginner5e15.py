"""Deprecated alias for :func:`morie.fn.gaussian_approx_biased`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gaussian_approx_biased import gaussian_approx_biased as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_15"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_15(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.gaussian_approx_biased` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_15() is the book-coordinate name for gaussian_approx_biased(); "
        "it will be removed. Use morie.fn.gaussian_approx_biased() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
