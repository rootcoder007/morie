"""Deprecated alias for :func:`morie.fn.likelihood_ratio_chi2`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .likelihood_ratio_chi2 import likelihood_ratio_chi2 as _impl

__all__ = ["ca_chapter_4_equation_18"]


def ca_chapter_4_equation_18(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.likelihood_ratio_chi2` instead."""
    warnings.warn(
        "ca_chapter_4_equation_18() is the book-coordinate name for likelihood_ratio_chi2(); "
        "it will be removed. Use morie.fn.likelihood_ratio_chi2() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
