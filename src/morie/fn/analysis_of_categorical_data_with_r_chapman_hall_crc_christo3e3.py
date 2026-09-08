"""Deprecated alias for :func:`morie.fn.product_multinomial_pmf`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .product_multinomial_pmf import product_multinomial_pmf as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_3"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_3(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.product_multinomial_pmf` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_3_equation_3() is the book-coordinate name for product_multinomial_pmf(); "
        "it will be removed. Use morie.fn.product_multinomial_pmf() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
