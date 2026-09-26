"""
Zonal anisotropy variogram

Category: Variogram
"""


def vmzrn(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Zonal anisotropy variogram

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmzrn.vmzrn is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmzrn"
alias = "vmzrn"
quote = "He who has a why to live can bear almost any how. -- Friedrich Nietzsche"
vmzrn = vmzrn


def cheatsheet() -> str:
    return "vmzrn({}) -> Zonal anisotropy variogram"
