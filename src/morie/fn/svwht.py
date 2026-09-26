"""Wittman divergence model (policy-motivated)"""


def wittman_model(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Wittman divergence model (policy-motivated)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svwht.wittman_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


witt = wittman_model


def cheatsheet() -> str:
    return "wittman_model({}) -> Wittman divergence model (policy-motivated)"


# compact alias per ledger/NAMING.md
wittmanmodel = wittman_model
