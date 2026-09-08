"""Deprecated alias for :func:`morie.fn.noncentrality_delta_generic`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .noncentrality_delta_generic import noncentrality_delta_generic as _impl

__all__ = ["ca_chapter_8_equation_1"]


def ca_chapter_8_equation_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.noncentrality_delta_generic` instead."""
    warnings.warn(
        "ca_chapter_8_equation_1() is the book-coordinate name for noncentrality_delta_generic(); "
        "it will be removed. Use morie.fn.noncentrality_delta_generic() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
