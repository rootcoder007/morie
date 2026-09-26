"""U-learner for CATE."""

__all__ = ["u_learner"]


def u_learner(y, D, X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    U-learner for CATE

    Formula: residualized + unconfoundedness moment

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
        "morie.fn.ulrnir.u_learner is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "ulrnir: U-learner for CATE"


# compact alias per ledger/NAMING.md
ulearner = u_learner
