"""Deprecated alias for :func:`morie.fn.hypergeometric_pmf`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .hypergeometric_pmf import hypergeometric_pmf as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_71"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_71(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.hypergeometric_pmf` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_71() is the book-coordinate name for hypergeometric_pmf(); "
        "it will be removed. Use morie.fn.hypergeometric_pmf() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
