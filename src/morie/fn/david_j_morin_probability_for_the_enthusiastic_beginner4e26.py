"""Deprecated alias for :func:`morie.fn.exponential_waiting_density`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .exponential_waiting_density import exponential_waiting_density as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_26"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_26(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.exponential_waiting_density` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_4_equation_26() is the book-coordinate name for exponential_waiting_density(); "
        "it will be removed. Use morie.fn.exponential_waiting_density() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
