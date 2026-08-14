# morie.fn -- wave2 slice x_2_01 (rootcoder007/morie)
"""Samejima graded response model -- alias of :mod:`morie.fn.grmsam`.

Samejima (1969), "Estimation of latent ability using a response pattern
of graded scores", Psychometrika Monograph Supplement 34(4, Pt. 2),
doi:10.1007/BF03372160.

Already implemented in :mod:`morie.fn.grmsam` as
``graded_response_samejima``; this module re-exports it under the
shorter name rather than carrying a second copy.
"""

from __future__ import annotations

from .grmsam import _grm_probs, graded_response_samejima  # noqa: F401

graded_response = graded_response_samejima

__all__ = ["graded_response", "graded_response_samejima"]


def cheatsheet():
    return "irtgrm: Samejima graded response model -- alias of grmsam"

# public names resolved by fn/_lazy_map.json
gradedresponse = graded_response_samejima
