"""Simulate from a bivariate extreme-value copula."""

__all__ = ["evt_bv_evd_sim"]


def evt_bv_evd_sim(alpha, n):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Simulate from a bivariate extreme-value copula

    Formula: x_i = -1/log(U_i^{α_i}) per Stephenson

    Parameters
    ----------
    alpha : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x, y

    References
    ----------
    Stephenson (2003)
    """
    raise NotImplementedError(
        "morie.fn.evbevsim.evt_bv_evd_sim is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "evbevsim: Simulate from a bivariate extreme-value copula"


# compact alias per ledger/NAMING.md
evtbvevdsim = evt_bv_evd_sim
