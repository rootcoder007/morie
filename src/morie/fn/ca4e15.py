"""Deprecated alias for :func:`morie.fn.wald_statistic`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .wald_statistic import wald_statistic as _impl

__all__ = ["ca_chapter_4_equation_15"]


def ca_chapter_4_equation_15(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.wald_statistic` instead."""
    warnings.warn(
        "ca_chapter_4_equation_15() is the book-coordinate name for wald_statistic(); "
        "it will be removed. Use morie.fn.wald_statistic() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
