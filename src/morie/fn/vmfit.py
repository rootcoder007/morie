"""
Variogram model fitting (WLS)

Category: Variogram
"""


def vmfit(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram model fitting (WLS)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmfit.vmfit is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmfit"
alias = "vmfit"
quote = "Mathematics is the queen of the sciences. -- Carl Friedrich Gauss"
vmfit = vmfit


def cheatsheet() -> str:
    return "vmfit({}) -> Variogram model fitting (WLS)"
