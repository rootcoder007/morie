"""1D ideal point estimation"""


def ideal_point_1d(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    1D ideal point estimation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svip1.ideal_point_1d is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


idea = ideal_point_1d


def cheatsheet() -> str:
    return "ideal_point_1d({}) -> 1D ideal point estimation"


# compact alias per ledger/NAMING.md
idealpoint1d = ideal_point_1d
