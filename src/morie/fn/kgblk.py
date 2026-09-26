# morie.fn -- function file (rootcoder007/morie)
"""Block kriging prediction"""


def block_kriging(values, x, y=None, *, model="spherical"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Block kriging prediction

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.kgblk.block_kriging is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


bloc = block_kriging


def cheatsheet() -> str:
    return "block_kriging({}) -> Block kriging prediction"


# compact alias per ledger/NAMING.md
blockkriging = block_kriging
