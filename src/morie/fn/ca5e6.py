"""Deprecated alias for :func:`morie.fn.cumulative_probability`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cumulative_probability import cumulative_probability as _impl

__all__ = ["ca_chapter_5_equation_6"]


def ca_chapter_5_equation_6(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cumulative_probability` instead."""
    warnings.warn(
        "ca_chapter_5_equation_6() is the book-coordinate name for cumulative_probability(); "
        "it will be removed. Use morie.fn.cumulative_probability() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
