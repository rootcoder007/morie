"""Deprecated alias for :func:`morie.fn.multinomial_probs`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .multinomial_probs import multinomial_probs as _impl

__all__ = ["ca_chapter_5_equation_3"]


def ca_chapter_5_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.multinomial_probs` instead."""
    warnings.warn(
        "ca_chapter_5_equation_3() is the book-coordinate name for multinomial_probs(); "
        "it will be removed. Use morie.fn.multinomial_probs() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
