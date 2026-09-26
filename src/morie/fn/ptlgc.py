# morie.fn -- function file (rootcoder007/morie)
"""Log-Gaussian Cox process"""


def log_gaussian_cox(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Log-Gaussian Cox process

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptlgc.log_gaussian_cox is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


log_ = log_gaussian_cox


def cheatsheet() -> str:
    return "log_gaussian_cox({}) -> Log-Gaussian Cox process"


# compact alias per ledger/NAMING.md
loggaussiancox = log_gaussian_cox
