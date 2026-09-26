"""
Variogram envelope bootstrap

Category: Variogram
"""


def vmenv(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram envelope bootstrap

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmenv.vmenv is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmenv"
alias = "vmenv"
quote = "The heart has its reasons of which reason knows nothing. -- Blaise Pascal"
vmenv = vmenv


def cheatsheet() -> str:
    return "vmenv({}) -> Variogram envelope bootstrap"
