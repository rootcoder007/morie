# morie.fn -- function file (rootcoder007/morie)
"""
Ordinary kriging filter

Category: KrigFilt
"""


def okflt(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Ordinary kriging filter

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.okflt.okflt is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "okflt"
alias = "okflt"
quote = "To understand God's thoughts we must study statistics. -- Florence Nightingale"
okflt = okflt


def cheatsheet() -> str:
    return "okflt({}) -> Ordinary kriging filter"
