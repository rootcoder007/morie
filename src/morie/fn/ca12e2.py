"""Deprecated alias for :func:`morie.fn.morans_i_expected`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .morans_i_expected import morans_i_expected as _impl

__all__ = ["ca_chapter_12_equation_2"]


def ca_chapter_12_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.morans_i_expected` instead."""
    warnings.warn(
        "ca_chapter_12_equation_2() is the book-coordinate name for morans_i_expected(); "
        "it will be removed. Use morie.fn.morans_i_expected() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
