# morie.fn -- function file (rootcoder007/morie)
"""Party split model.

Category: Spatial
"""

from . import _array_core as np

from ._containers import DescriptiveResult


def mpspl(data=None, n=50):
    """Party split model.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal(n)
    stat = float(np.mean(data))
    return DescriptiveResult(
        name=short,
        value=stat,
        extra={"n": len(data), "mean": float(np.mean(data)), "std": float(np.std(data))},
    )


short = "mpspl"
alias = "mpspl"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
mpspl = mpspl


def cheatsheet() -> str:
    return "mpspl({}) -> Party split model."
