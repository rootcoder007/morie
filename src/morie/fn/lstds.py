# morie.fn -- function file (rootcoder007/morie)
"""List all datasets with cache status."""

from morie.data import list_datasets as _fn

lstds = _fn
list_datasets = _fn


def cheatsheet() -> str:
    return "lstds() -> List all datasets with cache status."
