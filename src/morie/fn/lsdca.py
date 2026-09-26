"""Robust LDA via MCD class covariances."""

__all__ = ["robust_lda"]


def robust_lda(X, y):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Robust LDA via MCD class covariances

    Formula: replace μ_k, Σ_k with MCD versions

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
    Croux-Dehon (2001)
    """
    raise NotImplementedError(
        "morie.fn.lsdca.robust_lda is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "lsdca: Robust LDA via MCD class covariances"


# compact alias per ledger/NAMING.md
robustlda = robust_lda
