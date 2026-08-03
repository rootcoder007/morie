"""Deprecated alias for :func:`morie.fn.cumulative_logit`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cumulative_logit import cumulative_logit as _impl

__all__ = ["ca_chapter_5_equation_7"]


def ca_chapter_5_equation_7(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cumulative_logit` instead."""
    warnings.warn(
        "ca_chapter_5_equation_7() is the book-coordinate name for cumulative_logit(); "
        "it will be removed. Use morie.fn.cumulative_logit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
