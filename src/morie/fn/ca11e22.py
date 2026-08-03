"""Deprecated alias for :func:`morie.fn.d_from_r_pointbiserial`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .d_from_r_pointbiserial import d_from_r_pointbiserial as _impl

__all__ = ["ca_chapter_11_equation_22"]


def ca_chapter_11_equation_22(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.d_from_r_pointbiserial` instead."""
    warnings.warn(
        "ca_chapter_11_equation_22() is the book-coordinate name for d_from_r_pointbiserial(); "
        "it will be removed. Use morie.fn.d_from_r_pointbiserial() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
