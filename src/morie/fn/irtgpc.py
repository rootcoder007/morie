# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Generalized partial credit model (GPCM) -- alias of :mod:`morie.fn.gpcm`.

Muraki (1992), "A generalized partial credit model: application of an
EM algorithm", Applied Psychological Measurement 16(2):159-176,
doi:10.1177/014662169201600206.

The method is already implemented in :mod:`morie.fn.gpcm` under the
same public name ``generalized_partial_credit``; this module re-exports
it rather than carrying a second copy of the same estimator.
"""

from __future__ import annotations

from .gpcm import _gpcm_probs, generalized_partial_credit  # noqa: F401

__all__ = ["generalized_partial_credit"]


def cheatsheet():
    return "irtgpc: Generalized partial credit model (GPCM) -- alias of gpcm"
