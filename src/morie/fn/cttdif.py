"""CTT item difficulty."""

__all__ = ["ctt_difficulty"]


def ctt_difficulty(X):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    CTT item difficulty

    Formula: p_j = mean(X_j)

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nunnally-Bernstein (1994)
    """
    raise NotImplementedError(
        "morie.fn.cttdif.ctt_difficulty is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "cttdif: CTT item difficulty"


# compact alias per ledger/NAMING.md
cttdifficulty = ctt_difficulty
