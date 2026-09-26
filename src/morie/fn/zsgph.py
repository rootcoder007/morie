"""GP hyperparameter optimization"""


def gp_hyperparams(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    GP hyperparameter optimization

    Returns
    -------
    SpatialResult
    """
    raise NotImplementedError(
        "morie.fn.zsgph.gp_hyperparams is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


gp_h = gp_hyperparams


def cheatsheet() -> str:
    return "gp_hyperparams({}) -> GP hyperparameter optimization"


# compact alias per ledger/NAMING.md
gphyperparams = gp_hyperparams
