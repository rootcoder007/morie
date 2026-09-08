"""Deprecated alias for :func:`morie.fn.cox_snell_r2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cox_snell_r2 import cox_snell_r2 as _impl

__all__ = ["ca_chapter_4_equation_13"]


def ca_chapter_4_equation_13(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cox_snell_r2` instead."""
    warnings.warn(
        "ca_chapter_4_equation_13() is the book-coordinate name for cox_snell_r2(); "
        "it will be removed. Use morie.fn.cox_snell_r2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
