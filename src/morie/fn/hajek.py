# morie.fn -- function file (hadesllm/morie)
"""Hajek estimator for population mean."""

from morie.survey import hajek_mean as _fn

hajek = _fn
hajek_mean = _fn


def cheatsheet() -> str:
    return 'hajek() -> Hajek estimator for population mean.'
