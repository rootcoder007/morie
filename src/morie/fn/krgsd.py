# morie.fn -- function file (rootcoder007/morie)
"""
Kriging standard deviation

Category: KrigFilt
"""


def krgsd(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Kriging standard deviation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.krgsd.krgsd is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "krgsd"
alias = "krgsd"
quote = "Mathematics is the queen of the sciences. -- Carl Friedrich Gauss"
krgsd = krgsd


def cheatsheet() -> str:
    return "krgsd({}) -> Kriging standard deviation"
