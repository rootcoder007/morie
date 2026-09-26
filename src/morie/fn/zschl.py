"""Cholesky spatial simulation"""


def chol_sim(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Cholesky spatial simulation

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zschl.chol_sim is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


chol = chol_sim


def cheatsheet() -> str:
    return "chol_sim({}) -> Cholesky spatial simulation"


# compact alias per ledger/NAMING.md
cholsim = chol_sim
