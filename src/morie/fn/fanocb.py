"""Fano's inequality on error probability."""

__all__ = ["fano_inequality"]


def fano_inequality(pe, X_card):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Fano's inequality on error probability

    Formula: H(X|Y) <= H(P_e) + P_e log(|X|-1)

    Parameters
    ----------
    pe : array-like
        Input data.
    X_card : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fano (1961)
    """
    raise NotImplementedError(
        "morie.fn.fanocb.fano_inequality is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "fanocb: Fano's inequality on error probability"


# compact alias per ledger/NAMING.md
fanoinequality = fano_inequality
