"""
Variogram model fitting (MLE)

Category: Variogram
"""


def vmfml(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Variogram model fitting (MLE)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmfml.vmfml is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmfml"
alias = "vmfml"
quote = "Luck is what happens when preparation meets opportunity. -- Seneca"
vmfml = vmfml


def cheatsheet() -> str:
    return "vmfml({}) -> Variogram model fitting (MLE)"
