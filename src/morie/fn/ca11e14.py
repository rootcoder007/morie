"""Deprecated alias for :func:`morie.fn.r_from_fisher_z`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .r_from_fisher_z import r_from_fisher_z as _impl

__all__ = ["ca_chapter_11_equation_14"]


def ca_chapter_11_equation_14(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.r_from_fisher_z` instead."""
    warnings.warn(
        "ca_chapter_11_equation_14() is the book-coordinate name for r_from_fisher_z(); "
        "it will be removed. Use morie.fn.r_from_fisher_z() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
