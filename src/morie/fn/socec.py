"""
Cation exchange capacity

Category: SoilSp
"""


def socec(data=None, depth=None, coords=None, n=50):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Cation exchange capacity

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.socec.socec is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


short = "socec"
alias = "socec"
quote = "It is not the strongest that survives, but the most adaptable. -- Charles Darwin"
socec = socec


def cheatsheet() -> str:
    return "socec({}) -> Cation exchange capacity"
