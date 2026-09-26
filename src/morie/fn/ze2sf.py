"""Two-step floating catchment area"""


def two_step_fca(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Two-step floating catchment area

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.ze2sf.two_step_fca is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


two_ = two_step_fca


def cheatsheet() -> str:
    return "two_step_fca({}) -> Two-step floating catchment area"


# compact alias per ledger/NAMING.md
twostepfca = two_step_fca
