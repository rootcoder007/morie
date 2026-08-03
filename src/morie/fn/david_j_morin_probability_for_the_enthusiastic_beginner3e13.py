"""Deprecated alias for :func:`morie.fn.expectation_linear`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .expectation_linear import expectation_linear as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_13"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_13(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.expectation_linear` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_3_equation_13() is the book-coordinate name for expectation_linear(); "
        "it will be removed. Use morie.fn.expectation_linear() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
