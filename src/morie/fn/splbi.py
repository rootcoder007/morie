"""
Bicubic spline interpolation

Category: KrigFilt
"""


def splbi(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Bicubic spline interpolation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.splbi.splbi is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "splbi"
alias = "splbi"
quote = "The only true wisdom is in knowing you know nothing. -- Socrates"
splbi = splbi


def cheatsheet() -> str:
    return "splbi({}) -> Bicubic spline interpolation"
