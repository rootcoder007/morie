"""Deprecated alias for :func:`morie.fn.se_g`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .se_g import se_g as _impl

__all__ = ["ca_chapter_11_equation_7"]


def ca_chapter_11_equation_7(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.se_g` instead."""
    warnings.warn(
        "ca_chapter_11_equation_7() is the book-coordinate name for se_g(); "
        "it will be removed. Use morie.fn.se_g() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
