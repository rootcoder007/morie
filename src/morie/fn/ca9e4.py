"""Deprecated alias for :func:`morie.fn.chi2_2x2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .chi2_2x2 import chi2_2x2 as _impl

__all__ = ["ca_chapter_9_equation_4"]


def ca_chapter_9_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.chi2_2x2` instead."""
    warnings.warn(
        "ca_chapter_9_equation_4() is the book-coordinate name for chi2_2x2(); "
        "it will be removed. Use morie.fn.chi2_2x2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
