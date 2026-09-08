"""Deprecated alias for :func:`morie.fn.posterior_density_binomial`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .posterior_density_binomial import posterior_density_binomial as _impl

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_23"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_23(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.posterior_density_binomial` instead."""
    warnings.warn(
        "analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_23() is the book-coordinate name for posterior_density_binomial(); "
        "it will be removed. Use morie.fn.posterior_density_binomial() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
