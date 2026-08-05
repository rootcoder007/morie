"""Deprecated alias for :func:`morie.fn.basisexp`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .basisexp import basisexp as _impl

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_2_equation_30"]


def bookadvanced_elementsofstatisticallearning_chapter_2_equation_30(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.basisexp` instead."""
    warnings.warn(
        "bookadvanced_elementsofstatisticallearning_chapter_2_equation_30() is the book-coordinate name for basisexp(); "
        "it will be removed. Use morie.fn.basisexp() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning2e30: deprecated alias for basisexp"
