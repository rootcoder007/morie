"""K-nearest neighbors weights"""


def w_knn(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    K-nearest neighbors weights

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrwkn.w_knn is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


w_k = w_knn


def cheatsheet() -> str:
    return "w_knn({}) -> K-nearest neighbors weights"


# compact alias per ledger/NAMING.md
wknn = w_knn
