"""Deprecated alias for :func:`morie.fn.binomial_pmf_vector`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .binomial_pmf_vector import binomial_pmf_vector as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_10"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_10(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.binomial_pmf_vector` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_10() is the book-coordinate name for binomial_pmf_vector(); "
        "it will be removed. Use morie.fn.binomial_pmf_vector() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
