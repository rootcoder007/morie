"""Deprecated alias for :func:`morie.fn.gaussian_approx_2n`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gaussian_approx_2n import gaussian_approx_2n as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_13"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_13(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.gaussian_approx_2n` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_13() is the book-coordinate name for gaussian_approx_2n(); "
        "it will be removed. Use morie.fn.gaussian_approx_2n() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
