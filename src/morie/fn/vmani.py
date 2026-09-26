"""
Anisotropic variogram

Category: Variogram
"""


def vmani(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Anisotropic variogram

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmani.vmani is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmani"
alias = "vmani"
quote = "A journey of a thousand miles begins with a single step. -- Lao Tzu"
vmani = vmani


def cheatsheet() -> str:
    return "vmani({}) -> Anisotropic variogram"
