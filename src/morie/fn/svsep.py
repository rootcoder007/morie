"""Separating hyperplane"""


def separating_hyp(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Separating hyperplane

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svsep.separating_hyp is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


sepa = separating_hyp


def cheatsheet() -> str:
    return "separating_hyp({}) -> Separating hyperplane"


# compact alias per ledger/NAMING.md
separatinghyp = separating_hyp
