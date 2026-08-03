"""Deprecated alias for :func:`morie.fn.noncentrality_delta_r`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .noncentrality_delta_r import noncentrality_delta_r as _impl

__all__ = ["ca_chapter_8_equation_6"]


def ca_chapter_8_equation_6(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.noncentrality_delta_r` instead."""
    warnings.warn(
        "ca_chapter_8_equation_6() is the book-coordinate name for noncentrality_delta_r(); "
        "it will be removed. Use morie.fn.noncentrality_delta_r() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
