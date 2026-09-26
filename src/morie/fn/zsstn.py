"""Non-separable space-time covariance"""


def st_cov_nonsep(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Non-separable space-time covariance

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsstn.st_cov_nonsep is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


st_c = st_cov_nonsep


def cheatsheet() -> str:
    return "st_cov_nonsep({}) -> Non-separable space-time covariance"


# compact alias per ledger/NAMING.md
stcovnonsep = st_cov_nonsep
