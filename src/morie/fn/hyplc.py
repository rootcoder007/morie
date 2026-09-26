"""Harmonic mean marginal likelihood (cautionary)."""

__all__ = ["harmonic_mean_estimator"]


def harmonic_mean_estimator(log_lik):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Harmonic mean marginal likelihood (cautionary)

    Formula: m(y) ≈ ( (1/S) sum_s 1/p(y|theta_s) )^{-1}

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
    Newton & Raftery (1994); cf. Neal's pathological
    """
    raise NotImplementedError(
        "morie.fn.hyplc.harmonic_mean_estimator is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "hyplc: Harmonic mean marginal likelihood (cautionary)"
