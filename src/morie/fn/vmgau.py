"""
Gaussian variogram model

Category: Variogram
"""


def vmgau(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Gaussian variogram model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmgau.vmgau is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmgau"
alias = "vmgau"
quote = "If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton"
vmgau = vmgau


def cheatsheet() -> str:
    return "vmgau({}) -> Gaussian variogram model"
