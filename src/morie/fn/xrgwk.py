"""GWR kernel function"""


def gwr_kernel(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GWR kernel function

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrgwk.gwr_kernel is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gwr_ = gwr_kernel


def cheatsheet() -> str:
    return "gwr_kernel({}) -> GWR kernel function"


# compact alias per ledger/NAMING.md
gwrkernel = gwr_kernel
