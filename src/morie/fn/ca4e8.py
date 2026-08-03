"""Deprecated alias for :func:`morie.fn.odds_ratio_unit_change`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .odds_ratio_unit_change import odds_ratio_unit_change as _impl

__all__ = ["ca_chapter_4_equation_8"]


def ca_chapter_4_equation_8(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.odds_ratio_unit_change` instead."""
    warnings.warn(
        "ca_chapter_4_equation_8() is the book-coordinate name for odds_ratio_unit_change(); "
        "it will be removed. Use morie.fn.odds_ratio_unit_change() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
