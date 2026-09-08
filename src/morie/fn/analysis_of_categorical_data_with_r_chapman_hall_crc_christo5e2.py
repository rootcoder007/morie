"""Deprecated alias for :func:`morie.fn.bic_posterior_probs`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .bic_posterior_probs import bic_posterior_probs as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_equation_2"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_equation_2(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.bic_posterior_probs` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_5_equation_2() is the book-coordinate name for bic_posterior_probs(); "
        "it will be removed. Use morie.fn.bic_posterior_probs() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
