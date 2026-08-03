"""Deprecated alias for :func:`morie.fn.cohens_d_sample`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cohens_d_sample import cohens_d_sample as _impl

__all__ = ["ca_chapter_11_equation_1"]


def ca_chapter_11_equation_1(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cohens_d_sample` instead."""
    warnings.warn(
        "ca_chapter_11_equation_1() is the book-coordinate name for cohens_d_sample(); "
        "it will be removed. Use morie.fn.cohens_d_sample() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
