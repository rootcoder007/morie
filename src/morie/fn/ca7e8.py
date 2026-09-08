"""Deprecated alias for :func:`morie.fn.lr_test_chi2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .lr_test_chi2 import lr_test_chi2 as _impl

__all__ = ["ca_chapter_7_equation_8"]


def ca_chapter_7_equation_8(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.lr_test_chi2` instead."""
    warnings.warn(
        "ca_chapter_7_equation_8() is the book-coordinate name for lr_test_chi2(); "
        "it will be removed. Use morie.fn.lr_test_chi2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
