"""Deprecated alias for :func:`morie.fn.ptotal`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .ptotal import ptotal as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_55"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_55(p_a, p_z_given_a, p_z_given_not_a):
    """Deprecated; use :func:`morie.fn.ptotal` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_55() is the book-coordinate name for ptotal(); "
        "it will be removed. Use morie.fn.ptotal() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl([float(p_a), 1.0 - float(p_a)], [p_z_given_a, p_z_given_not_a])
