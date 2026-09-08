"""Deprecated alias for :func:`morie.fn.negative_binomial_variance`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .negative_binomial_variance import negative_binomial_variance as _impl

__all__ = ["ca_chapter_6_equation_8"]


def ca_chapter_6_equation_8(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.negative_binomial_variance` instead."""
    warnings.warn(
        "ca_chapter_6_equation_8() is the book-coordinate name for negative_binomial_variance(); "
        "it will be removed. Use morie.fn.negative_binomial_variance() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
