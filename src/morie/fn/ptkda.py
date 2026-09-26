# morie.fn -- function file (rootcoder007/morie)
"""Adaptive kernel density"""


def kde_adaptive(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Adaptive kernel density

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ptkda.kde_adaptive is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


kde_ = kde_adaptive


def cheatsheet() -> str:
    return "kde_adaptive({}) -> Adaptive kernel density"


# compact alias per ledger/NAMING.md
kdeadaptive = kde_adaptive
