# morie.fn -- function file (rootcoder007/morie)
"""Homogeneous Poisson point process"""


def poisson_process(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Homogeneous Poisson point process

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptpoi.poisson_process is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


pois = poisson_process


def cheatsheet() -> str:
    return "poisson_process({}) -> Homogeneous Poisson point process"


# compact alias per ledger/NAMING.md
poissonprocess = poisson_process
