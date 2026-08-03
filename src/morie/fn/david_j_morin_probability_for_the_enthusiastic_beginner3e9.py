"""Deprecated alias for :func:`morie.fn.joint_independent`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .joint_independent import joint_independent as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_9"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_9(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.joint_independent` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_9() is the book-coordinate name for joint_independent(); "
        "it will be removed. Use morie.fn.joint_independent() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
