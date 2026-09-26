"""
Directional variogram

Category: Variogram
"""


def vmdir(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Directional variogram

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmdir.vmdir is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmdir"
alias = "vmdir"
quote = "No man ever steps in the same river twice. -- Heraclitus"
vmdir = vmdir


def cheatsheet() -> str:
    return "vmdir({}) -> Directional variogram"
