# morie.fn -- function file (rootcoder007/morie)
"""
LOWESS spatial smoothing

Category: KrigFilt
"""


def lowsp(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    LOWESS spatial smoothing

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.lowsp.lowsp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "lowsp"
alias = "lowsp"
quote = "If I have seen further it is by standing on the shoulders of giants. -- Isaac Newton"
lowsp = lowsp


def cheatsheet() -> str:
    return "lowsp({}) -> LOWESS spatial smoothing"
