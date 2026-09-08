"""Deprecated alias for :func:`morie.fn.beta_standardized`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .beta_standardized import beta_standardized as _impl

__all__ = ["ca_chapter_2_equation_20"]


def ca_chapter_2_equation_20(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.beta_standardized` instead."""
    warnings.warn(
        "ca_chapter_2_equation_20() is the book-coordinate name for beta_standardized(); "
        "it will be removed. Use morie.fn.beta_standardized() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
