"""Robust LM test for error"""


def lm_robust_error(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Robust LM test for error

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.xrlmb.lm_robust_error is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


lm_r = lm_robust_error


def cheatsheet() -> str:
    return "lm_robust_error({}) -> Robust LM test for error"


# compact alias per ledger/NAMING.md
lmrobusterror = lm_robust_error
