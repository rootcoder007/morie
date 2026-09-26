"""
Semivariance windowed temporal

Category: Variogram
"""


def vmswt(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Semivariance windowed temporal

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmswt.vmswt is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmswt"
alias = "vmswt"
quote = "Errors using inadequate data are much less than those using none. -- Charles Babbage"
vmswt = vmswt


def cheatsheet() -> str:
    return "vmswt({}) -> Semivariance windowed temporal"
