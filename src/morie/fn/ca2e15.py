"""Deprecated alias for :func:`morie.fn.adjusted_r2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .adjusted_r2 import adjusted_r2 as _impl

__all__ = ["ca_chapter_2_equation_15"]


def ca_chapter_2_equation_15(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.adjusted_r2` instead."""
    warnings.warn(
        "ca_chapter_2_equation_15() is the book-coordinate name for adjusted_r2(); "
        "it will be removed. Use morie.fn.adjusted_r2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
