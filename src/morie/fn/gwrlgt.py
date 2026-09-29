# morie.fn -- function file (rootcoder007/morie)
"""GWR logistic regression (binary outcome)."""

from .gwrpois import _ggwr


def gwrlgt(y, X, coords, bw=0.5, kernel="bisquare", adaptive=False, tol=1e-10, maxiter=200):
    r"""Geographically weighted logistic regression by the GWmodel local scoring.

    For a binary ``y`` the working response ``z_j = eta_j + (y_j - mu_j) /
    (mu_j (1 - mu_j))`` is regressed at each location by WLS with weights
    ``w_ij mu_j (1 - mu_j)`` (unit weights and ``mu = 0.5`` at the start),
    ``eta_i = x_i beta_i`` updated and the loop stopped on the relative change
    of the Bernoulli log-likelihood (Fotheringham, Brunsdon and Charlton
    2002, ch. 8; Nakaya et al. 2005). ``tol=1e-5, maxiter=20`` reproduces
    ``GWmodel::ggwr.basic(family = "binomial")``.

    References
    ----------
    Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2002).
    *Geographically Weighted Regression*. Wiley.
    Nakaya, T., Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2005).
    Geographically weighted Poisson regression for disease association
    mapping. *Statistics in Medicine* 24, 2695-2717.

    Examples
    --------
    >>> P = [(float(i % 5), float(i // 5)) for i in range(20)]
    >>> X = [[(0.37 * i) % 1.3] for i in range(20)]
    >>> y = [float((i * 7) % 3 == 0) for i in range(20)]
    >>> r = gwrlgt(y, X, P, 6.0, kernel="gaussian")
    >>> round(r["betas"][0][1], 8)
    -1.16439525
    """
    return _ggwr(y, X, coords, bw, kernel, adaptive, tol, maxiter, "binomial")


gwrlgt_fn = gwrlgt


def cheatsheet() -> str:
    return "gwrlgt(y, X, coords, bw) -> GW logistic regression by local scoring (GWmodel::ggwr.basic)."
