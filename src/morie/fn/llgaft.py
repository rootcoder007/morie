"""Log-logistic AFT model."""

__all__ = ["log_logistic_aft"]


def log_logistic_aft(time, event, X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Log-logistic AFT model

    Formula: log(T) = beta'X + sigma * Z, Z ~ logistic(0,1)

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kalbfleisch & Prentice (2002) §2.2.5
    """
    raise NotImplementedError(
        "morie.fn.llgaft.log_logistic_aft is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "llgaft: Log-logistic AFT model"


# compact alias per ledger/NAMING.md
loglogisticaft = log_logistic_aft
