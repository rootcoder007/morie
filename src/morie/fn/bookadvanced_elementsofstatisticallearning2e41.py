"""Deprecated alias for :func:`morie.fn.nadwat`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .nadwat import nadwat as _impl

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_2_equation_41"]


def bookadvanced_elementsofstatisticallearning_chapter_2_equation_41(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.nadwat` instead."""
    warnings.warn(
        "bookadvanced_elementsofstatisticallearning_chapter_2_equation_41() is the book-coordinate name for nadwat(); "
        "it will be removed. Use morie.fn.nadwat() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning2e41: deprecated alias for nadwat"
