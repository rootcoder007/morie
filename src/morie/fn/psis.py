"""PSIS-LOO importance weight smoothing."""

__all__ = ["pareto_smoothed_importance_sampling"]


def pareto_smoothed_importance_sampling(log_lik):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    PSIS-LOO importance weight smoothing

    Formula: w_i^(s) = clip( p(y_i | theta_s)^{-1}, GPD smooth )

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
    Vehtari, Simpson, Gelman, Yao, Gabry (2024)
    """
    raise NotImplementedError(
        "morie.fn.psis.pareto_smoothed_importance_sampling is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "psis: PSIS-LOO importance weight smoothing"
