"""
Stable variogram model

Category: Variogram
"""


def vmstb(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Stable variogram model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmstb.vmstb is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmstb"
alias = "vmstb"
quote = "The Analytical Engine weaves algebraic patterns. -- Ada Lovelace"
vmstb = vmstb


def cheatsheet() -> str:
    return "vmstb({}) -> Stable variogram model"
