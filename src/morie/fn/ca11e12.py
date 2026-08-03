"""Deprecated alias for :func:`morie.fn.fisher_z`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .fisher_z import fisher_z as _impl

__all__ = ["ca_chapter_11_equation_12"]


def ca_chapter_11_equation_12(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.fisher_z` instead."""
    warnings.warn(
        "ca_chapter_11_equation_12() is the book-coordinate name for fisher_z(); "
        "it will be removed. Use morie.fn.fisher_z() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
