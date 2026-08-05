"""Deprecated alias for :func:`morie.fn.nnet1lay`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .nnet1lay import nnet1lay as _impl

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_2_equation_45"]


def bookadvanced_elementsofstatisticallearning_chapter_2_equation_45(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.nnet1lay` instead."""
    warnings.warn(
        "bookadvanced_elementsofstatisticallearning_chapter_2_equation_45() is the book-coordinate name for nnet1lay(); "
        "it will be removed. Use morie.fn.nnet1lay() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning2e45: deprecated alias for nnet1lay"
