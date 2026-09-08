"""Deprecated alias for :func:`morie.fn.bimid`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bimid import bimid as _impl

__all__ = ["wilcox_chapter_2_equation_12"]


def wilcox_chapter_2_equation_12(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.bimid` instead."""
    warnings.warn(
        "wilcox_chapter_2_equation_12() is the book-coordinate name for bimid(); "
        "it will be removed. Use morie.fn.bimid() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "wilcox2e12: deprecated alias for bimid"
