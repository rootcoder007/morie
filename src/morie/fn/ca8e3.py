"""Deprecated alias for :func:`morie.fn.power_from_delta_t`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .power_from_delta_t import power_from_delta_t as _impl

__all__ = ["ca_chapter_8_equation_3"]


def ca_chapter_8_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.power_from_delta_t` instead."""
    warnings.warn(
        "ca_chapter_8_equation_3() is the book-coordinate name for power_from_delta_t(); "
        "it will be removed. Use morie.fn.power_from_delta_t() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
