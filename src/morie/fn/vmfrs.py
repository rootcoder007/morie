"""
Variogram model fitting (REML)

Category: Variogram
"""


def vmfrs(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram model fitting (REML)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmfrs.vmfrs is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmfrs"
alias = "vmfrs"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
vmfrs = vmfrs


def cheatsheet() -> str:
    return "vmfrs({}) -> Variogram model fitting (REML)"
