"""
Wave variogram model

Category: Variogram
"""


def vmwav(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Wave variogram model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmwav.vmwav is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmwav"
alias = "vmwav"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
vmwav = vmwav


def cheatsheet() -> str:
    return "vmwav({}) -> Wave variogram model"
