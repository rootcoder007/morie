"""Deprecated alias for :func:`morie.fn.olsfit`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .olsfit import olsfit as _impl

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_2_equation_26"]


def bookadvanced_elementsofstatisticallearning_chapter_2_equation_26(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.olsfit` instead."""
    warnings.warn(
        "bookadvanced_elementsofstatisticallearning_chapter_2_equation_26() is the book-coordinate name for olsfit(); "
        "it will be removed. Use morie.fn.olsfit() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning2e26: deprecated alias for olsfit"
