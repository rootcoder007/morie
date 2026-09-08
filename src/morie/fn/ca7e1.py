"""Deprecated alias for :func:`morie.fn.grand_mean_model`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .grand_mean_model import grand_mean_model as _impl

__all__ = ["ca_chapter_7_equation_1"]


def ca_chapter_7_equation_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.grand_mean_model` instead."""
    warnings.warn(
        "ca_chapter_7_equation_1() is the book-coordinate name for grand_mean_model(); "
        "it will be removed. Use morie.fn.grand_mean_model() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
