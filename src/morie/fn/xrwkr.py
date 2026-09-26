"""Kernel weights"""


def w_kernel(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kernel weights

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrwkr.w_kernel is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


w_ke = w_kernel


def cheatsheet() -> str:
    return "w_kernel({}) -> Kernel weights"


# compact alias per ledger/NAMING.md
wkernel = w_kernel
