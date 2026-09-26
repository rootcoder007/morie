"""PySR symbolic regression."""

__all__ = ["pysr_regression"]


def pysr_regression(X, y):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    PySR symbolic regression

    Formula: genetic programming + Pareto front

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cranmer (2023) PySR
    """
    raise NotImplementedError(
        "morie.fn.pysrSR.pysr_regression is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "pysrSR: PySR symbolic regression"


# compact alias per ledger/NAMING.md
pysrregression = pysr_regression
