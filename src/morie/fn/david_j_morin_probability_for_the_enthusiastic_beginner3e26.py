"""Deprecated alias for :func:`morie.fn.var_sum_with_cov`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .var_sum_with_cov import var_sum_with_cov as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_26"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_26(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.var_sum_with_cov` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_26() is the book-coordinate name for var_sum_with_cov(); "
        "it will be removed. Use morie.fn.var_sum_with_cov() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
