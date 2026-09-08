"""Deprecated alias for :func:`morie.fn.multinomial_conditional_or`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .multinomial_conditional_or import multinomial_conditional_or as _impl

__all__ = ["ca_chapter_5_equation_4"]


def ca_chapter_5_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.multinomial_conditional_or` instead."""
    warnings.warn(
        "ca_chapter_5_equation_4() is the book-coordinate name for multinomial_conditional_or(); "
        "it will be removed. Use morie.fn.multinomial_conditional_or() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
