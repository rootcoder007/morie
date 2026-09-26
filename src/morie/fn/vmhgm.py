"""
Hagstrom variogram estimator

Category: Variogram
"""


def vmhgm(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Hagstrom variogram estimator

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmhgm.vmhgm is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmhgm"
alias = "vmhgm"
quote = "In the midst of chaos, there is also opportunity. -- Sun Tzu"
vmhgm = vmhgm


def cheatsheet() -> str:
    return "vmhgm({}) -> Hagstrom variogram estimator"
