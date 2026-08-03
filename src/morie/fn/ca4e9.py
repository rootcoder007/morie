"""Deprecated alias for :func:`morie.fn.derivative_at_mean`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .derivative_at_mean import derivative_at_mean as _impl

__all__ = ["ca_chapter_4_equation_9"]


def ca_chapter_4_equation_9(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.derivative_at_mean` instead."""
    warnings.warn(
        "ca_chapter_4_equation_9() is the book-coordinate name for derivative_at_mean(); "
        "it will be removed. Use morie.fn.derivative_at_mean() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
