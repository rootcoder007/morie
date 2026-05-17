# morie.fn -- function file (hadesllm/morie)
"""Double ML."""

from morie.fn.dml import estimate_double_ml

douml = estimate_double_ml
double_ml = estimate_double_ml


def cheatsheet() -> str:
    return 'douml() -> Double ML.'
