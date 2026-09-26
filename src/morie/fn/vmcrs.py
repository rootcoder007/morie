"""
Cross-variogram

Category: Variogram
"""


def vmcrs(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Cross-variogram

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmcrs.vmcrs is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmcrs"
alias = "vmcrs"
quote = "The Analytical Engine weaves algebraic patterns. -- Ada Lovelace"
vmcrs = vmcrs


def cheatsheet() -> str:
    return "vmcrs({}) -> Cross-variogram"
