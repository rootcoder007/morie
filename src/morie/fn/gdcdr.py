# morie.fn -- function file (rootcoder007/morie)
"""
Crude death rate spatial

Category: GeoDem
"""


def gdcdr(population=None, births=None, deaths=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Crude death rate spatial

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.gdcdr.gdcdr is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "gdcdr"
alias = "gdcdr"
quote = "Measure what is measurable, and make measurable what is not. -- Galileo Galilei"
gdcdr = gdcdr


def cheatsheet() -> str:
    return "gdcdr({}) -> Crude death rate spatial"
