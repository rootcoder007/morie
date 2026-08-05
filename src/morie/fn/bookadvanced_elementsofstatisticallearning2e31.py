"""Deprecated alias for :func:`morie.fn.sigbasis`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .sigbasis import sigbasis as _impl

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_2_equation_31"]


def bookadvanced_elementsofstatisticallearning_chapter_2_equation_31(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.sigbasis` instead."""
    warnings.warn(
        "bookadvanced_elementsofstatisticallearning_chapter_2_equation_31() is the book-coordinate name for sigbasis(); "
        "it will be removed. Use morie.fn.sigbasis() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning2e31: deprecated alias for sigbasis"
