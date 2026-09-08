"""Deprecated alias for :func:`morie.fn.gxe_multitrait_model`.

The book-coordinate name is kept so existing code keeps working.  It warns
once and forwards to the method-named function.
"""

import warnings

from .gxe_multitrait_model import gxe_multitrait_model as _impl

__all__ = ["mvsml_linear_mixed_models_eq_5_6"]


def mvsml_linear_mixed_models_eq_5_6(*args, **kwargs):
    """Deprecated; use :func:`morie.fn.gxe_multitrait_model` instead."""
    warnings.warn(
        "mvsml_linear_mixed_models_eq_5_6() is the book-coordinate name for gxe_multitrait_model(); "
        "it will be removed. Use morie.fn.gxe_multitrait_model() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _impl(*args, **kwargs)
