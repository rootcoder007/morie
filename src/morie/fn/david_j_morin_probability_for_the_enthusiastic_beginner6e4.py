"""Deprecated alias for :func:`morie.fn.muy`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .muy import muy as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_4"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_4(m, mu_x, mu_z):
    """Deprecated; use :func:`morie.fn.muy` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_4() is the book-coordinate name for muy(); "
        "it will be removed. Use morie.fn.muy() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(m, mu_x, mu_z)
