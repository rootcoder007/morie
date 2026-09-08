"""Deprecated alias for :func:`morie.fn.ols_matrix`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ols_matrix import ols_matrix as _impl

__all__ = ["ca_chapter_12_equation_3"]


def ca_chapter_12_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.ols_matrix` instead."""
    warnings.warn(
        "ca_chapter_12_equation_3() is the book-coordinate name for ols_matrix(); "
        "it will be removed. Use morie.fn.ols_matrix() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
