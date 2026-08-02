# morie.fn -- function file (rootcoder007/morie)
"""Empirical Bayes: estimate hyperparameter alpha from marginal likelihood."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ghosal_empirical_bayes_np"]


def ghosal_empirical_bayes_np(x, alpha_grid=None, sigma=1.0):
    r"""Empirical-Bayes choice of the Dirichlet concentration
    (Ghosal Ch. 4-6):

    .. math:: \hat\alpha = \arg\max_\alpha
              \int p_\alpha(X)\,d\Pi_\alpha(\theta),

    the maximiser of the MARGINAL likelihood, with the
    infinite-dimensional parameter integrated out.

    For a Dirichlet process the marginal is available in closed form
    through the Polya urn: the joint law of the sample factorises
    into terms ``alpha/(alpha+i-1)`` for each NEW value and
    ``n_k/(alpha+i-1)`` for a repeat, so the marginal likelihood
    depends on the data only through the sequence of distinct values.
    That makes :math:`\hat\alpha` a function of the number of
    CLUSTERS, and the intuition is exact: many distinct values push
    :math:`\hat\alpha` up, few push it down.

    Empirical Bayes is not free. Plugging :math:`\hat\alpha` back
    in treats an estimated hyperparameter as known, so the resulting
    credible sets can under-cover; the book's fully Bayesian
    alternative puts a prior on :math:`\alpha`.
    ``understates_uncertainty`` records that rather than leaving it
    implicit.

    Parameters
    ----------
    x : array-like
        Observations. Ties define the clusters the marginal depends
        on; continuous data are binned to the observed resolution.
    alpha_grid : array-like, optional
        Concentrations searched over.
    sigma : float > 0
        Retained for interface symmetry with the mixture modules.

    Returns
    -------
    RichResult
        keys: ``alpha_hat``, ``alpha_grid``, ``log_marginal``,
        ``n_clusters``, ``understates_uncertainty`` (True),
        ``fully_bayes_alternative``, ``n``, ``method``.
    References
    ----------
    Ghosal and van der Vaart, Ch. 4 (Dirichlet process marginals)
    and Ch. 6; the Polya urn factorisation is Sec. 4.1.4.
    """
    from ._sci_core import gammaln

    xv = np.asarray(x, dtype=float).ravel()
    n = xv.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}.")
    _, counts = np.unique(xv, return_counts=True)
    k = counts.size
    ag = np.logspace(-2, 2, 200) if alpha_grid is None else \
        np.atleast_1d(np.asarray(alpha_grid, dtype=float))
    if np.any(ag <= 0):
        raise ValueError("alpha values must be positive.")
    # log marginal of the partition under the Polya urn (Antoniak):
    # k log alpha + log Gamma(alpha) - log Gamma(alpha + n) + const
    lm = k * np.log(ag) + gammaln(ag) - gammaln(ag + n)
    j = int(np.argmax(lm))
    return RichResult(payload={
        "alpha_hat": float(ag[j]), "alpha_grid": ag, "log_marginal": lm,
        "n_clusters": int(k),
        "understates_uncertainty": True,
        "fully_bayes_alternative": "put a prior on alpha and integrate it out",
        "n": int(n),
        "method": "Empirical Bayes for the DP concentration; the marginal depends on the CLUSTER count"})


def cheatsheet():
    return "gh_emp_bayes: alpha-hat is driven by the number of clusters; plugging it in under-covers"
