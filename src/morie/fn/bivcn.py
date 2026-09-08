# morie.fn -- function file (rootcoder007/morie)
"""Bivariate causal direction via independence of residuals -- front-end."""

from __future__ import annotations

from .anmod import additive_noise_model as _anm

__all__ = ["bivariate_causal_test"]


def bivariate_causal_test(X, Y, regressor=None, cdf=None, B=200, seed=None):
    """Bivariate causal direction by HSIC on additive-noise residuals.

    The same procedure as :func:`morie.fn.anmod.additive_noise_model`,
    which holds the implementation: fit both directions, test each
    residual against its putative cause with HSIC, and report the
    direction whose residual looks independent.

    ``regressor`` is accepted for signature compatibility and ignored.
    Only the Nadaraya-Watson smoother is implemented, and silently
    accepting the name of a different regressor while using that one
    would misreport what was run.

    See :func:`morie.fn.anmod.additive_noise_model` for the method, its
    identifiability limits and the reference.
    """
    if regressor not in (None, "nw", "nadaraya-watson"):
        raise ValueError(
            f"regressor={regressor!r} is not implemented; only the Nadaraya-Watson "
            "smoother is available. Pass None."
        )
    del cdf  # significance comes from the permutation null, not a supplied CDF
    return _anm(X, Y, B=B, seed=seed)


def cheatsheet():
    return "bivcn: bivariate causal direction test; see anmod"
