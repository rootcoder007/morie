"""Deprecated alias for :func:`morie.fn.gausstail`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gausstail import gausstail as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_25"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_25(n_sigmas=20.0, sigma=1.0):
    """Deprecated; use :func:`morie.fn.gausstail` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_5_equation_25() is the book-coordinate name for gausstail(); "
        "it will be removed. Use morie.fn.gausstail() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(n_sigmas, sigma)
