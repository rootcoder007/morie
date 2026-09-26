"""Effective parameters from WAIC (p_WAIC)."""

__all__ = ["effective_parameters_waic"]


def effective_parameters_waic(log_lik):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Effective parameters from WAIC (p_WAIC)

    Formula: p_WAIC = sum_i Var_s( log p(y_i | theta_s) )

    Parameters
    ----------
    log_lik : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vehtari, Gelman, Gabry (2017)
    """
    raise NotImplementedError(
        "morie.fn.pwaic.effective_parameters_waic is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "pwaic: Effective parameters from WAIC (p_WAIC)"
