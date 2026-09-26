"""
Dissolved oxygen water

Category: WtrQual
"""


def wqdo(data=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Dissolved oxygen water

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.wqdo.wqdo is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "wqdo"
alias = "wqdo"
quote = "The whole is greater than the sum of its parts. -- Aristotle"
wqdo = wqdo


def cheatsheet() -> str:
    return "wqdo({}) -> Dissolved oxygen water"
