"""Robust LM test for lag"""


def lm_robust_lag(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Robust LM test for lag

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrlmr.lm_robust_lag is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


lm_r = lm_robust_lag


def cheatsheet() -> str:
    return "lm_robust_lag({}) -> Robust LM test for lag"


# compact alias per ledger/NAMING.md
lmrobustlag = lm_robust_lag
