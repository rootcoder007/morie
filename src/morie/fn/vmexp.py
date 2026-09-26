"""
Exponential variogram model

Category: Variogram
"""


def vmexp(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Exponential variogram model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmexp.vmexp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmexp"
alias = "vmexp"
quote = "Statistics is the grammar of science. -- Karl Pearson"
vmexp = vmexp


def cheatsheet() -> str:
    return "vmexp({}) -> Exponential variogram model"
