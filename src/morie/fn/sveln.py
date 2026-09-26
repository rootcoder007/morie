"""Elbow method for dimensions"""


def elbow_spatial(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Elbow method for dimensions

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.sveln.elbow_spatial is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


elbo = elbow_spatial


def cheatsheet() -> str:
    return "elbow_spatial({}) -> Elbow method for dimensions"


# compact alias per ledger/NAMING.md
elbowspatial = elbow_spatial
