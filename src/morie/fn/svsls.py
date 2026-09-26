"""Issue salience weighted model"""


def salience_model(data, *, method="default"):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.

    Issue salience weighted model

    Returns
    -------
    DescriptiveResult
    """
    raise NotImplementedError(
        "morie.fn.svsls.salience_model is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


sali = salience_model


def cheatsheet() -> str:
    return "salience_model({}) -> Issue salience weighted model"


# compact alias per ledger/NAMING.md
saliencemodel = salience_model
