"""Deprecated alias for :func:`morie.fn.fixed_effect_weight`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .fixed_effect_weight import fixed_effect_weight as _impl

__all__ = ["ca_chapter_11_equation_34"]


def ca_chapter_11_equation_34(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.fixed_effect_weight` instead."""
    warnings.warn(
        "ca_chapter_11_equation_34() is the book-coordinate name for fixed_effect_weight(); "
        "it will be removed. Use morie.fn.fixed_effect_weight() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
