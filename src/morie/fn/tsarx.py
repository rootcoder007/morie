"""
Spatial ARX model

Category: TempSpat
"""


def tsarx(data=None, coords=None, times=None, n=50, t=10):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial ARX model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.tsarx.tsarx is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "tsarx"
alias = "tsarx"
quote = "The heart has its reasons of which reason knows nothing. -- Blaise Pascal"
tsarx = tsarx


def cheatsheet() -> str:
    return "tsarx({}) -> Spatial ARX model"
