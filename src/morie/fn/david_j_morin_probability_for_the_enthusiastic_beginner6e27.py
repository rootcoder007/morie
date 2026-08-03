"""Deprecated alias for :func:`morie.fn.prediction_improvement`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .prediction_improvement import prediction_improvement as _impl

__all__ = ["david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_27"]


def david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_27(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.prediction_improvement` instead."""
    warnings.warn(
        "david_j_morin_probability_for_the_enthusiastic_beginner_chapter_6_equation_27() is the book-coordinate name for prediction_improvement(); "
        "it will be removed. Use morie.fn.prediction_improvement() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
