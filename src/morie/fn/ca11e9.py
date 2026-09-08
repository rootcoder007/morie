"""Deprecated alias for :func:`morie.fn.se_log_rr`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .se_log_rr import se_log_rr as _impl

__all__ = ["ca_chapter_11_equation_9"]


def ca_chapter_11_equation_9(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.se_log_rr` instead."""
    warnings.warn(
        "ca_chapter_11_equation_9() is the book-coordinate name for se_log_rr(); "
        "it will be removed. Use morie.fn.se_log_rr() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
