"""Deprecated alias for :func:`morie.fn.joint_density_factorizes`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .joint_density_factorizes import joint_density_factorizes as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_64"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_64(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.joint_density_factorizes` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_64() is the book-coordinate name for joint_density_factorizes(); "
        "it will be removed. Use morie.fn.joint_density_factorizes() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
