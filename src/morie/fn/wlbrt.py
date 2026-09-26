"""
BRT species model

Category: WildlSp
"""


def wlbrt(abundance=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    BRT species model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.wlbrt.wlbrt is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "wlbrt"
alias = "wlbrt"
quote = "Measure what is measurable, and make measurable what is not. -- Galileo Galilei"
wlbrt = wlbrt


def cheatsheet() -> str:
    return "wlbrt({}) -> BRT species model"
