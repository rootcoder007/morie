"""Deprecated alias for :func:`morie.fn.noncentrality_lambda_f`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .noncentrality_lambda_f import noncentrality_lambda_f as _impl

__all__ = ["ca_chapter_8_equation_5"]


def ca_chapter_8_equation_5(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.noncentrality_lambda_f` instead."""
    warnings.warn(
        "ca_chapter_8_equation_5() is the book-coordinate name for noncentrality_lambda_f(); "
        "it will be removed. Use morie.fn.noncentrality_lambda_f() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
