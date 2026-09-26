"""
Matern variogram model

Category: Variogram
"""


def vmmat(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Matern variogram model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmmat.vmmat is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmmat"
alias = "vmmat"
quote = "Give me a place to stand and I will move the earth. -- Archimedes"
vmmat = vmmat


def cheatsheet() -> str:
    return "vmmat({}) -> Matern variogram model"
