"""Deprecated alias for :func:`morie.fn.tau2_dersimonian_laird`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .tau2_dersimonian_laird import tau2_dersimonian_laird as _impl

__all__ = ["ca_chapter_11_equation_44"]


def ca_chapter_11_equation_44(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.tau2_dersimonian_laird` instead."""
    warnings.warn(
        "ca_chapter_11_equation_44() is the book-coordinate name for tau2_dersimonian_laird(); "
        "it will be removed. Use morie.fn.tau2_dersimonian_laird() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
