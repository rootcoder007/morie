"""T-learner for CATE."""

__all__ = ["t_learner"]


def t_learner(y, D, X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    T-learner for CATE

    Formula: separate Y(1) and Y(0) models; tau = Y(1)-Y(0)

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Künzel et al (2019)
    """
    raise NotImplementedError(
        "morie.fn.tlearn.t_learner is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "tlearn: T-learner for CATE"


# compact alias per ledger/NAMING.md
tlearner = t_learner
