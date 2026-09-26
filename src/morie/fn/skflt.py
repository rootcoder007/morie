"""
Simple kriging filter

Category: KrigFilt
"""


def skflt(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Simple kriging filter

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.skflt.skflt is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "skflt"
alias = "skflt"
quote = "What is now proved was once only imagined. -- William Blake"
skflt = skflt


def cheatsheet() -> str:
    return "skflt({}) -> Simple kriging filter"
