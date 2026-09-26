"""Bliss point estimation"""


def bliss_point(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Bliss point estimation

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svblp.bliss_point is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


blis = bliss_point


def cheatsheet() -> str:
    return "bliss_point({}) -> Bliss point estimation"


# compact alias per ledger/NAMING.md
blisspoint = bliss_point
