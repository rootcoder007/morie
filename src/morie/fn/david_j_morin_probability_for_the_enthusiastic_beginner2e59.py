"""Deprecated alias for :func:`morie.fn.bayesexp`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bayesexp import bayesexp as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_59"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_59(p_a, p_z_given_a, p_z_given_not_a):
    """Deprecated; use :func:`morie.fn.bayesexp` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_2_equation_59() is the book-coordinate name for bayesexp(); "
        "it will be removed. Use morie.fn.bayesexp() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(p_a, p_z_given_a, p_z_given_not_a)
