"""Deprecated alias for :func:`morie.fn.snp_blup_gebv`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .snp_blup_gebv import snp_blup_gebv as _impl

__all__ = ["mvsml_preprocessing_eq_2_4"]


def mvsml_preprocessing_eq_2_4(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.snp_blup_gebv` instead."""
    warnings.warn(
        "mvsml_preprocessing_eq_2_4() is the book-coordinate name for snp_blup_gebv(); "
        "it will be removed. Use morie.fn.snp_blup_gebv() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
