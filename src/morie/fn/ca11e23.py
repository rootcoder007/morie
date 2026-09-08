"""Deprecated alias for :func:`morie.fn.se_d_from_se_r`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .se_d_from_se_r import se_d_from_se_r as _impl

__all__ = ["ca_chapter_11_equation_23"]


def ca_chapter_11_equation_23(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.se_d_from_se_r` instead."""
    warnings.warn(
        "ca_chapter_11_equation_23() is the book-coordinate name for se_d_from_se_r(); "
        "it will be removed. Use morie.fn.se_d_from_se_r() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
