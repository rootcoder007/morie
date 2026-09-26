"""GP kernel selection"""


def gp_kernel(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GP kernel selection

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsgpk.gp_kernel is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gp_k = gp_kernel


def cheatsheet() -> str:
    return "gp_kernel({}) -> GP kernel selection"


# compact alias per ledger/NAMING.md
gpkernel = gp_kernel
