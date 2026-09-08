"""Deprecated alias for :func:`morie.fn.random_effects_weight`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .random_effects_weight import random_effects_weight as _impl

__all__ = ["ca_chapter_11_equation_43"]


def ca_chapter_11_equation_43(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.random_effects_weight` instead."""
    warnings.warn(
        "ca_chapter_11_equation_43() is the book-coordinate name for random_effects_weight(); "
        "it will be removed. Use morie.fn.random_effects_weight() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
