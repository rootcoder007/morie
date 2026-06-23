"""Simple random sample from a DataFrame."""

from morie.sampling import simple_random_sample as _fn

srs = _fn
simple_random_sample = _fn


def cheatsheet() -> str:
    return "srs() -> Simple random sample from a DataFrame."
