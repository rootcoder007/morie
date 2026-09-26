"""Covariance matrix from variogram"""


def covario_matrix(coords, values, *, nbins=15):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Covariance matrix from variogram

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.vgcvr.covario_matrix is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


cova = covario_matrix


def cheatsheet() -> str:
    return "covario_matrix({}) -> Covariance matrix from variogram"


# compact alias per ledger/NAMING.md
covariomatrix = covario_matrix
