"""
Thin-plate spline interpolation

Category: KrigFilt
"""


def spltps(x=None, y=None, values=None, grid_size=20, range_param=30.0, sill=1.0, nugget=0.1):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Thin-plate spline interpolation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.spltps.spltps is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "spltps"
alias = "spltps"
quote = "The whole is greater than the sum of its parts. -- Aristotle"
spltps = spltps


def cheatsheet() -> str:
    return "spltps({}) -> Thin-plate spline interpolation"
