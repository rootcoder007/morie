"""BayesB sparse marker prior."""

__all__ = ["bayes_b_marker"]


def bayes_b_marker(y, M, pi):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    BayesB sparse marker prior

    Formula: sigma_j^2=0 with prob pi else scaled-inv-chi2

    Parameters
    ----------
    y : array-like
        Input data.
    M : array-like
        Input data.
    pi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Meuwissen-Hayes-Goddard (2001)
    """
    raise NotImplementedError(
        "morie.fn.baysbm.bayes_b_marker is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "baysbm: BayesB sparse marker prior"


# compact alias per ledger/NAMING.md
bayesbmarker = bayes_b_marker
