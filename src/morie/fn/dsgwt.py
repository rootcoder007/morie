# morie.fn -- function file (rootcoder007/morie)
"""Inverse-probability design weights for stratified samples."""

from morie.sampling import compute_design_weights as _fn

dsgwt = _fn
compute_design_weights = _fn


def cheatsheet() -> str:
    return 'dsgwt() -> Inverse-probability design weights for stratified samples.'
