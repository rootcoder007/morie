"""Deprecated alias for :func:`morie.fn.cohens_d_population`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cohens_d_population import cohens_d_population as _impl

__all__ = ["ca_chapter_8_equation_2"]


def ca_chapter_8_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cohens_d_population` instead."""
    warnings.warn(
        "ca_chapter_8_equation_2() is the book-coordinate name for cohens_d_population(); "
        "it will be removed. Use morie.fn.cohens_d_population() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
