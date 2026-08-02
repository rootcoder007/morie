# SPDX-License-Identifier: AGPL-3.0-or-later
"""Besag-York-Mollie convolution model for disease mapping."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_glmm import (bym_identifiability_note, bym_map,
                          bym_median_log_prior, neighbour_structure, smr)

__all__ = ["schabenberger_bym"]


def schabenberger_bym(counts, expected, adjacency, kappa, lam,
                      max_iter=200, tol=1e-11):
    """Fit the BYM convolution model, Besag, York and Mollie (1991) Sec. 4.

    For i = 1, ..., n areas, with y_i the observed count and c_i the
    expected count,

        y_i | x_i ~ Poisson(c_i exp{x_i}),      x_i = u_i + v_i

    where u carries spatially structured variation under the intrinsic
    autoregression (4.2) with scale kappa, and v carries unstructured
    heterogeneity under an exchangeable Gaussian prior with scale lambda.
    The relative risk in area i is exp{u_i + v_i}, and the convolution of
    the two components is what the model contributes: Schabenberger gives
    them separately at (6.104) and (6.102) but never their sum.

    The paper's reading of the two scales: "kappa tends to 0 implies
    constant u_i's, whereas kappa large implies correspondingly large but
    spatially structured variation. Similarly, lambda tends to 0 implies
    v = 0, whereas lambda large implies substantial but unstructured
    extra-Poisson variability."

    Returned are the conditional MAP estimates u*, v* given kappa and
    lambda, obtained by Newton on the log of the joint posterior (4.5).
    That is well posed rather than merely convenient: the paper states the
    log posterior in u and v is "a strictly concave differentiable function
    of u and v and therefore possesses a single maximum", so there is one
    optimum and no multi-start question.

    TWO IDENTITIES ARE REPORTED AND SHOULD BE CHECKED. Sec. 4 gives
    sum_i v*_i = 0 and sum_i c_i exp{u*_i + v*_i} = sum_i y_i, "so that the
    fitted total number of cases matches the observed total". Neither is
    imposed here; both fall out of stationarity, because the structure
    matrix has zero row sums. They are returned as ``sum_v`` and the
    ``fitted_total`` / ``observed_total`` pair precisely so that a fit which
    failed to reach a stationary point is visible rather than silent.

    IDENTIFIABILITY. Only u + v enters the likelihood, so the data cannot
    separate the two variance components. kappa and lambda are therefore
    arguments here, not estimated: fully Bayesian treatment needs the
    hyperprior (4.6) and a sampler, and the empirical-Bayes shortcut of
    fixing them "will on average produce shorter but erroneous interval
    estimates ... because it does not account for variability in the
    estimation of the hyperparameters". For an identifiable alternative see
    the Leroux LCAR prior, which nests the exchangeable and intrinsic cases
    in a single parameter.

    Parameters
    ----------
    counts : array-like, shape (n,)
        Observed counts y_i.
    expected : array-like, shape (n,)
        Expected counts c_i, typically age-standardised. Must be positive.
    adjacency : array-like, shape (n, n)
        Symmetric 0/1 contiguity matrix with a zero diagonal.
    kappa, lam : float
        Scales of the structured and unstructured components. The paper's
        own estimates were kappa = 0.129, lambda = 0.011 for thyroid cancer
        across the 94 departements of France, and kappa = lambda = 0.009 for
        multiple myeloma.

    Returns
    -------
    RichResult
        Keys: ``u``, ``v``, ``x``, ``relative_risk``, ``fitted``, ``smr``,
        ``sum_v``, ``fitted_total``, ``observed_total``, ``log_posterior``,
        ``converged``, ``n_iter``, ``identifiability``.

    References
    ----------
    Besag, York and Mollie (1991), Ann. Inst. Statist. Math. 43(1):1-59,
    Sec. 4, eqs (4.2)-(4.6); Schabenberger Ch 6, Sec 6.4.3.2
    """
    y = np.asarray(counts, dtype=float).ravel()
    c = np.asarray(expected, dtype=float).ravel()
    fit = bym_map(y, c, adjacency, kappa, lam, max_iter=max_iter, tol=tol)

    payload = dict(fit)
    payload["smr"] = smr(y, c)
    payload["kappa"] = float(kappa)
    payload["lam"] = float(lam)
    payload["identifiability"] = bym_identifiability_note()
    payload["n_neighbours"] = np.asarray(adjacency, dtype=float).sum(axis=1)
    payload["median_log_prior"] = bym_median_log_prior(fit["u"], adjacency,
                                                       kappa)
    payload["shrinkage"] = float(
        np.std(np.log(payload["smr"] + 1e-12)) - np.std(fit["x"]))

    total_gap = abs(fit["fitted_total"] - fit["observed_total"])
    lines = [("areas", y.size), ("kappa", kappa), ("lambda", lam),
             ("iterations", fit["n_iter"]), ("converged", fit["converged"]),
             ("sum of v* (should be 0)", fit["sum_v"]),
             ("fitted total vs observed",
              "%.6f vs %.6f" % (fit["fitted_total"], fit["observed_total"]))]

    problems = []
    if not fit["converged"]:
        problems.append("Newton did not converge")
    if abs(fit["sum_v"]) > 1e-6:
        problems.append("sum of v* is %.3g, not 0" % fit["sum_v"])
    if total_gap > 1e-5 * max(fit["observed_total"], 1.0):
        problems.append("fitted total misses the observed total by %.3g"
                        % total_gap)
    if problems:
        payload["warning"] = (
            "the Sec. 4 stationarity identities do not hold, so this is not "
            "a maximum of (4.5): " + "; ".join(problems))

    return RichResult(title="Besag-York-Mollie convolution model",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spbym: Besag-York-Mollie convolution for disease mapping "
            "(BYM 1991 Sec. 4) -- ICAR u plus exchangeable v, MAP by Newton")
