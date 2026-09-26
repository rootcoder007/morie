"""Spatial loss function (quadratic/city-block)"""


def loss_function(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Spatial loss function (quadratic/city-block)

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svlss.loss_function is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


loss = loss_function


def cheatsheet() -> str:
    return "loss_function({}) -> Spatial loss function (quadratic/city-block)"


# compact alias per ledger/NAMING.md
lossfunction = loss_function
