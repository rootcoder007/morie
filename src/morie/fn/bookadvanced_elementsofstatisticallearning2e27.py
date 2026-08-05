"""Deprecated alias for :func:`morie.fn.epeols`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .epeols import epeols as _impl

__all__ = ["bookadvanced_elementsofstatisticallearning_chapter_2_equation_27"]


def bookadvanced_elementsofstatisticallearning_chapter_2_equation_27(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.epeols` instead."""
    warnings.warn(
        "bookadvanced_elementsofstatisticallearning_chapter_2_equation_27() is the book-coordinate name for epeols(); "
        "it will be removed. Use morie.fn.epeols() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)


def cheatsheet():
    return "bookadvanced_elementsofstatisticallearning2e27: deprecated alias for epeols"
