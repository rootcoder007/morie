"""Deprecated alias for :func:`morie.fn.variance_components_sigma2_u`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .variance_components_sigma2_u import variance_components_sigma2_u as _impl

__all__ = ["ca_chapter_7_equation_6"]


def ca_chapter_7_equation_6(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.variance_components_sigma2_u` instead."""
    warnings.warn(
        "ca_chapter_7_equation_6() is the book-coordinate name for variance_components_sigma2_u(); "
        "it will be removed. Use morie.fn.variance_components_sigma2_u() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
