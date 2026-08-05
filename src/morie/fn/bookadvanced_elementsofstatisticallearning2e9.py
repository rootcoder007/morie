"""Deprecated alias for :func:`morie.fn.epetheor`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .epetheor import epetheor as _impl

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_2_equation_9"]


def bookadvanced_elementsofstatisticallearning_chapter_2_equation_9(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.epetheor` instead."""
    warnings.warn(
        "bookadvanced_elementsofstatisticallearning_chapter_2_equation_9() is the book-coordinate name for epetheor(); "
        "it will be removed. Use morie.fn.epetheor() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning2e9: deprecated alias for epetheor"
