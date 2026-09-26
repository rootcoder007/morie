"""GP prediction"""


def gp_predict(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GP prediction

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsgpp.gp_predict is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gp_p = gp_predict


def cheatsheet() -> str:
    return "gp_predict({}) -> GP prediction"


# compact alias per ledger/NAMING.md
gppredict = gp_predict
