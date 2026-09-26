"""EM ideal point estimation"""


def ideal_point_em(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    EM ideal point estimation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svipe.ideal_point_em is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


idea = ideal_point_em


def cheatsheet() -> str:
    return "ideal_point_em({}) -> EM ideal point estimation"


# compact alias per ledger/NAMING.md
idealpointem = ideal_point_em
