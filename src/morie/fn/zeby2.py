"""BYM2 reparameterized model"""


def bym2_model(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    BYM2 reparameterized model

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zeby2.bym2_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


bym2 = bym2_model


def cheatsheet() -> str:
    return "bym2_model({}) -> BYM2 reparameterized model"


# compact alias per ledger/NAMING.md
bym2model = bym2_model
