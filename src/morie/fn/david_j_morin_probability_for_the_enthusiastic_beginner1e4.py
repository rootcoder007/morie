"""Deprecated alias for :func:`morie.fn.ordrep`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ordrep import ordrep as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_4"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_4(N, n):
    """Deprecated; use :func:`morie.fn.ordrep` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_1_equation_4() is the book-coordinate name for ordrep(); "
        "it will be removed. Use morie.fn.ordrep() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(N, n)
