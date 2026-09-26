"""S-learner for CATE."""

__all__ = ["s_learner"]


def s_learner(y, D, X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    S-learner for CATE

    Formula: single Y model with D as feature

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
        "morie.fn.slearn.s_learner is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "slearn: S-learner for CATE"


# compact alias per ledger/NAMING.md
slearner = s_learner
