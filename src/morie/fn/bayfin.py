"""Finite mixture model (Bayesian K-means)."""

__all__ = ["finite_mixture"]


def finite_mixture(y, K):
    """NOT IMPLEMENTED. The previous body of this callable did not compute the
    method named here (it returned a placeholder such as the mean of its
    input, or a statistic of internally generated data). It raises
    NotImplementedError until the real method is built.
    Finite mixture model (Bayesian K-means)

    Formula: y_i ~ sum_k pi_k N(mu_k, sigma_k); priors on mu, sigma, pi

    Parameters
    ----------
    y : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Diebolt-Robert (1994)
    """
    raise NotImplementedError(
        "morie.fn.bayfin.finite_mixture is not implemented yet: its former body returned a "
        "placeholder, not the method its name and docstring describe."
    )


def cheatsheet():
    return "bayfin: Finite mixture model (Bayesian K-means)"


# compact alias per ledger/NAMING.md
finitemixture = finite_mixture
