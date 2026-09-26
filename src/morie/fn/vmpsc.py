"""
Pseudo-cross-variogram

Category: Variogram
"""


def vmpsc(x=None, y=None, values=None, n_lags=15, max_lag=None):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Pseudo-cross-variogram

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.vmpsc.vmpsc is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "vmpsc"
alias = "vmpsc"
quote = "Statistics is the grammar of science. -- Karl Pearson"
vmpsc = vmpsc


def cheatsheet() -> str:
    return "vmpsc({}) -> Pseudo-cross-variogram"
