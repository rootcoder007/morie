"""BYM (Besag-York-Mollie) model"""


def bym_model(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    BYM (Besag-York-Mollie) model

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zebym.bym_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


bym_ = bym_model


def cheatsheet() -> str:
    return "bym_model({}) -> BYM (Besag-York-Mollie) model"


# compact alias per ledger/NAMING.md
bymmodel = bym_model
