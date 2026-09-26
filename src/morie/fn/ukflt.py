"""
Universal kriging filter

Category: KrigFilt
"""


def ukflt(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Universal kriging filter

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.ukflt.ukflt is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "ukflt"
alias = "ukflt"
quote = "Mathematics is the art of giving the same name to different things. -- Henri Poincare"
ukflt = ukflt


def cheatsheet() -> str:
    return "ukflt({}) -> Universal kriging filter"
