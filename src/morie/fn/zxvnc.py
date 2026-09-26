"""Vincenty geodesic distance"""


def vincenty_dist(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Vincenty geodesic distance

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.zxvnc.vincenty_dist is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


vinc = vincenty_dist


def cheatsheet() -> str:
    return "vincenty_dist({}) -> Vincenty geodesic distance"


# compact alias per ledger/NAMING.md
vincentydist = vincenty_dist
