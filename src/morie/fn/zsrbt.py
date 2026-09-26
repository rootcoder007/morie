"""Thin plate spline RBF"""


def rbf_thinplate(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Thin plate spline RBF

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsrbt.rbf_thinplate is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


rbf_ = rbf_thinplate


def cheatsheet() -> str:
    return "rbf_thinplate({}) -> Thin plate spline RBF"


# compact alias per ledger/NAMING.md
rbfthinplate = rbf_thinplate
