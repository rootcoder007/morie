"""Deprecated alias for :func:`morie.fn.d_probit`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .d_probit import d_probit as _impl

__all__ = ["ca_chapter_11_equation_20"]


def ca_chapter_11_equation_20(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.d_probit` instead."""
    warnings.warn(
        "ca_chapter_11_equation_20() is the book-coordinate name for d_probit(); "
        "it will be removed. Use morie.fn.d_probit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
