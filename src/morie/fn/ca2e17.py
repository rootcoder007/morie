"""Deprecated alias for :func:`morie.fn.f_overall_r2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .f_overall_r2 import f_overall_r2 as _impl

__all__ = ["ca_chapter_2_equation_17"]


def ca_chapter_2_equation_17(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.f_overall_r2` instead."""
    warnings.warn(
        "ca_chapter_2_equation_17() is the book-coordinate name for f_overall_r2(); "
        "it will be removed. Use morie.fn.f_overall_r2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
