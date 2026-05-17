# morie.fn -- function file (hadesllm/morie)
"""Ppssmp."""

from morie.sampling import pps_sample as _fn

ppssmp = _fn
pps_sample = _fn


def cheatsheet() -> str:
    return 'ppssmp() -> Ppssmp'
