"""
Variogram model selection (AIC)

Category: Variogram
"""


def vmsel(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram model selection (AIC)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmsel.vmsel is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmsel"
alias = "vmsel"
quote = "Give me a place to stand and I will move the earth. -- Archimedes"
vmsel = vmsel


def cheatsheet() -> str:
    return "vmsel({}) -> Variogram model selection (AIC)"
