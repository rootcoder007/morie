"""
Robust variogram (Cressie-Hawkins)

Category: Variogram
"""


def vmrob(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Robust variogram (Cressie-Hawkins)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmrob.vmrob is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmrob"
alias = "vmrob"
quote = "Number rules the universe. -- Pythagoras"
vmrob = vmrob


def cheatsheet() -> str:
    return "vmrob({}) -> Robust variogram (Cressie-Hawkins)"
