"""Deprecated alias for :func:`morie.fn.retestr`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .retestr import retestr as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_37"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_37(sigma_signal, sigma_noise):
    """Deprecated; use :func:`morie.fn.retestr` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_37() is the book-coordinate name for retestr(); "
        "it will be removed. Use morie.fn.retestr() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(sigma_signal, sigma_noise)
