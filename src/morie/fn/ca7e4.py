"""Deprecated alias for :func:`morie.fn.cluster_means_model`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .cluster_means_model import cluster_means_model as _impl

__all__ = ["ca_chapter_7_equation_4"]


def ca_chapter_7_equation_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.cluster_means_model` instead."""
    warnings.warn(
        "ca_chapter_7_equation_4() is the book-coordinate name for cluster_means_model(); "
        "it will be removed. Use morie.fn.cluster_means_model() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
