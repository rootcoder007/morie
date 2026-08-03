"""Deprecated alias for :func:`morie.fn.beta_posterior_pdf`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .beta_posterior_pdf import beta_posterior_pdf as _impl

__all__ = ["the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_24"]


def the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_24(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.beta_posterior_pdf` instead."""
    warnings.warn(
        "the_r_series_dick_j_brus_spatial_sampling_with_r_chapter_12_equation_24() is the book-coordinate name for beta_posterior_pdf(); "
        "it will be removed. Use morie.fn.beta_posterior_pdf() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
