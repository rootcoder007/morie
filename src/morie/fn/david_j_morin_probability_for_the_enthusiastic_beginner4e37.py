"""Deprecated alias for :func:`morie.fn.onemexp`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .onemexp import onemexp as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_37"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_37(lam_eps, n):
    """Deprecated; use :func:`morie.fn.onemexp` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_37() is the book-coordinate name for onemexp(); "
        "it will be removed. Use morie.fn.onemexp() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(lam_eps, n)
