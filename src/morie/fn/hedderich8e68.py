"""Deprecated alias for :func:`morie.fn.lrresid`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.

The body this name used to carry was NOT an implementation of Hedderich
eq. (8.68): the stub generator pasted the same one-sample
Kolmogorov-Smirnov test into every numbered-equation row of chapters 2-8,
so the old signature and the old numbers meant nothing.  The real
implementation, written from the rendered PDF page, is
:func:`morie.fn.lrresid`; the old signature is not preserved because there
was nothing behind it to preserve.
"""

import warnings

from .lrresid import lrresid as _impl

__all__ = ["hedderich_chapter_8_equation_68"]


def hedderich_chapter_8_equation_68(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.lrresid` instead."""
    warnings.warn(
        "hedderich_chapter_8_equation_68() is the book-coordinate name for lrresid(); "
        "it will be removed. Use morie.fn.lrresid() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
