"""
Field capacity soil

Category: SoilSp
"""


def sofcl(data=None, depth=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Field capacity soil

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.sofcl.sofcl is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "sofcl"
alias = "sofcl"
quote = "Measure what is measurable, and make measurable what is not. -- Galileo Galilei"
sofcl = sofcl


def cheatsheet() -> str:
    return "sofcl({}) -> Field capacity soil"
