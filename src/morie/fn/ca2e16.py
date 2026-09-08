"""Deprecated alias for :func:`morie.fn.f_overall_ss`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .f_overall_ss import f_overall_ss as _impl

__all__ = ["ca_chapter_2_equation_16"]


def ca_chapter_2_equation_16(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.f_overall_ss` instead."""
    warnings.warn(
        "ca_chapter_2_equation_16() is the book-coordinate name for f_overall_ss(); "
        "it will be removed. Use morie.fn.f_overall_ss() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
