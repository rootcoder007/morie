# morie.fn -- function file (rootcoder007/morie)
"""KDE bandwidth selection (spatial)"""


def kde_bandwidth(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    KDE bandwidth selection (spatial)

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptkdb.kde_bandwidth is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


kde_ = kde_bandwidth


def cheatsheet() -> str:
    return "kde_bandwidth({}) -> KDE bandwidth selection (spatial)"


# compact alias per ledger/NAMING.md
kdebandwidth = kde_bandwidth
