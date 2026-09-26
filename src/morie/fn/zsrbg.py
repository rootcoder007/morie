"""Gaussian RBF interpolation"""


def rbf_gaussian(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Gaussian RBF interpolation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsrbg.rbf_gaussian is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


rbf_ = rbf_gaussian


def cheatsheet() -> str:
    return "rbf_gaussian({}) -> Gaussian RBF interpolation"


# compact alias per ledger/NAMING.md
rbfgaussian = rbf_gaussian
