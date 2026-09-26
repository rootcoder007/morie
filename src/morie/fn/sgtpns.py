"""Perron-Frobenius leading eigenvalue + eigenvector."""

__all__ = ["sgt_perron_frobenius"]


def sgt_perron_frobenius(M):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Perron-Frobenius leading eigenvalue + eigenvector

    Formula: λ_PF = max{|λ_i|}; v_PF >= 0

    Parameters
    ----------
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lam_pf, v_pf

    References
    ----------
    Perron (1907); Frobenius (1908)
    """
    raise NotImplementedError(
        "morie.fn.sgtpns.sgt_perron_frobenius is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "sgtpns: Perron-Frobenius leading eigenvalue + eigenvector"
