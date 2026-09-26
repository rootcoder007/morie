"""
K-Bessel variogram model

Category: Variogram
"""


def vmkbs(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    K-Bessel variogram model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmkbs.vmkbs is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmkbs"
alias = "vmkbs"
quote = "Logic is the foundation of all certain knowledge. -- Leonhard Euler"
vmkbs = vmkbs


def cheatsheet() -> str:
    return "vmkbs({}) -> K-Bessel variogram model"
