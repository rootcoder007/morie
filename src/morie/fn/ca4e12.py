"""Deprecated alias for :func:`morie.fn.percent_correct_predictions`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .percent_correct_predictions import percent_correct_predictions as _impl

__all__ = ["ca_chapter_4_equation_12"]


def ca_chapter_4_equation_12(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.percent_correct_predictions` instead."""
    warnings.warn(
        "ca_chapter_4_equation_12() is the book-coordinate name for percent_correct_predictions(); "
        "it will be removed. Use morie.fn.percent_correct_predictions() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
