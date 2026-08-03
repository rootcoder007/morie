"""Deprecated alias for :func:`morie.fn.binomial_expansion`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binomial_expansion import binomial_expansion as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_21"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_21(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.binomial_expansion` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_21() is the book-coordinate name for binomial_expansion(); "
        "it will be removed. Use morie.fn.binomial_expansion() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
