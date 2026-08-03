"""Deprecated alias for :func:`morie.fn.odds_ratio_2x2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .odds_ratio_2x2 import odds_ratio_2x2 as _impl

__all__ = ["ca_chapter_11_equation_10"]


def ca_chapter_11_equation_10(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.odds_ratio_2x2` instead."""
    warnings.warn(
        "ca_chapter_11_equation_10() is the book-coordinate name for odds_ratio_2x2(); "
        "it will be removed. Use morie.fn.odds_ratio_2x2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
