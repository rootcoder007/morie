"""
Secchi depth spatial

Category: WtrQual
"""


def wqsec(data=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Secchi depth spatial

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.wqsec.wqsec is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "wqsec"
alias = "wqsec"
quote = "No man ever steps in the same river twice. -- Heraclitus"
wqsec = wqsec


def cheatsheet() -> str:
    return "wqsec({}) -> Secchi depth spatial"
