"""
Spectral density from variogram

Category: Variogram
"""


def vmspc(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spectral density from variogram

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmspc.vmspc is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmspc"
alias = "vmspc"
quote = "It does not matter how slowly you go as long as you do not stop. -- Confucius"
vmspc = vmspc


def cheatsheet() -> str:
    return "vmspc({}) -> Spectral density from variogram"
