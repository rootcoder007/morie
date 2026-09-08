"""Deprecated alias for :func:`morie.fn.se_fisher_z`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .se_fisher_z import se_fisher_z as _impl

__all__ = ["ca_chapter_11_equation_13"]


def ca_chapter_11_equation_13(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.se_fisher_z` instead."""
    warnings.warn(
        "ca_chapter_11_equation_13() is the book-coordinate name for se_fisher_z(); "
        "it will be removed. Use morie.fn.se_fisher_z() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
