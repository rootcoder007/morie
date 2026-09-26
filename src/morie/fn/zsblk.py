"""Spatial block bootstrap"""


def block_bootstrap(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial block bootstrap

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsblk.block_bootstrap is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


bloc = block_bootstrap


def cheatsheet() -> str:
    return "block_bootstrap({}) -> Spatial block bootstrap"


# compact alias per ledger/NAMING.md
blockbootstrap = block_bootstrap
