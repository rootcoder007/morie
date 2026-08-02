# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bayesian hierarchical spatial models for disease mapping."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_glmm import (interaction_structure, lcar_full_conditional,
                          lcar_precision, linear_trend_log_risk,
                          neighbour_structure, nonparametric_log_risk,
                          null_space_constraints, random_walk_structure, smr)

__all__ = ["schabenberger_bayes_hierarchical"]

_SPATIAL = ("exchangeable", "icar", "lcar")
_TEMPORAL = ("none", "rw1", "rw2")


def schabenberger_bayes_hierarchical(counts, expected, adjacency,
                                     spatial_prior="lcar", rho=0.5,
                                     n_time=None, temporal_prior="none",
                                     interaction=None, sigma2=1.0):
    """Specify a Bayesian hierarchical model for disease mapping, Sec. 6.4.

    The first stage is Clayton and Kaldor's (1987) relative-risk model,
    eq (6.99),

        [Z(s_i) | zeta(s_i)] ~ Poisson(E(s_i) zeta(s_i)),

    where the observed-to-expected ratio Z(s_i)/E(s_i) is the standardized
    mortality ratio and the maximum likelihood estimate of zeta(s_i).
    Covariates and random effects enter through eq (6.100),
    log{zeta(s_i)} = x(s_i)' beta + psi(s_i), giving (6.101).

    What distinguishes the models is the prior on psi.

    ``exchangeable``  eq (6.102), psi(s_i) iid G(0, sigma_psi^2). Adds
        excess but NOT spatially structured variation. The text calls the
        resulting smoothing "borrowing strength".
    ``icar``          eq (6.104) with adjacency weights: a conditionally
        specified prior whose structure matrix is singular, so the prior is
        improper and the covariance needs a pseudo-inverse.
    ``lcar``          the Leroux prior, Q = rho R + (1 - rho) I. This nests
        the other two -- rho = 0 is exchangeable, rho = 1 is intrinsic --
        so a single interpretable parameter replaces two variance
        components the data cannot separate. That identifiability failure is
        the reason to prefer it to the BYM convolution.

    Temporal and space-time structure follow the same pattern. A random walk
    of order 1 or 2 supplies the structured temporal effect; the
    interaction between space and time is one of the four Knorr-Held types,
    built as a Kronecker product of the two structure matrices.

    THE CONSTRAINT THAT MAKES INTERACTIONS IDENTIFIABLE. Every type except
    Type I has a rank-deficient structure matrix, and without constraints
    "the interaction terms are confounded with the main time effect". The
    remedy is to condition on A delta = 0, where the rows of A are the
    eigenvectors spanning the null space, and "the number of linear
    constraints which are necessary is always equal to the rank
    deficiency". Both are computed and returned, so the number of
    constraints is never guessed.

    This function specifies and diagnoses the model -- structure matrices,
    ranks, constraints, priors -- rather than running a sampler. Fitting
    these models needs MCMC or INLA, and the text warns that "convergence of
    MCMC algorithms can be very sensitive" to the hyperprior parameters.

    Parameters
    ----------
    counts, expected : array-like, shape (n,)
    adjacency : array-like, shape (n, n)
    spatial_prior : {"exchangeable", "icar", "lcar"}
    rho : float
        Leroux smoothing parameter in [0, 1]; used when ``spatial_prior``
        is "lcar".
    n_time : int, optional
        Number of periods; required for a temporal or interaction term.
    temporal_prior : {"none", "rw1", "rw2"}
    interaction : {"I", "II", "III", "IV"}, optional
    sigma2 : float

    Returns
    -------
    RichResult
        Keys: ``smr``, ``precision``, ``spatial_prior``, and when a temporal
        structure is present ``temporal_structure``, ``interaction_rank``,
        ``rank_deficiency``, ``n_constraints``, ``constraint_matrix``.

    References
    ----------
    Schabenberger Ch 6, Sec 6.4, eqs (6.99)-(6.104); Tonui, Mwalili and
    Wanjoya (2018), Open Journal of Statistics 8:811-830, eqs (6)-(12)
    """
    if spatial_prior not in _SPATIAL:
        raise ValueError(f"`spatial_prior` must be one of {_SPATIAL}")
    if temporal_prior not in _TEMPORAL:
        raise ValueError(f"`temporal_prior` must be one of {_TEMPORAL}")

    y = np.asarray(counts, dtype=float).ravel()
    e = np.asarray(expected, dtype=float).ravel()
    R = neighbour_structure(adjacency)
    n = R.shape[0]

    if spatial_prior == "exchangeable":
        Q = np.eye(n)
        note = "eq (6.102): aspatial, adds excess variation only"
    elif spatial_prior == "icar":
        Q = R
        note = ("eq (6.104): singular, so the prior is improper and the "
                "covariance requires a Moore-Penrose inverse")
    else:
        Q, _ = lcar_precision(R, rho, sigma2)
        note = (f"Leroux: rho = {rho}; rho = 0 is exchangeable, rho = 1 is "
                f"intrinsic, and unlike the BYM convolution it is "
                f"identifiable")

    payload = {"smr": smr(y, e), "precision": Q, "structure": R,
               "spatial_prior": spatial_prior, "prior_note": note,
               "n_areas": n, "sigma2": float(sigma2),
               "rank_deficiency_spatial": int(n - np.linalg.matrix_rank(Q))}
    lines = [("areas", n), ("spatial prior", spatial_prior)]
    if spatial_prior == "lcar":
        payload["rho"] = float(rho)
        lines.append(("rho", rho))

    if temporal_prior != "none" or interaction is not None:
        if n_time is None:
            raise ValueError("`n_time` is required for a temporal or "
                             "interaction structure")
        order = {"rw1": 1, "rw2": 2}.get(temporal_prior, 1)
        Rt = random_walk_structure(n_time, order)
        payload["temporal_structure"] = Rt
        payload["temporal_prior"] = temporal_prior
        payload["rank_deficiency_temporal"] = int(
            n_time - np.linalg.matrix_rank(Rt))
        lines += [("periods", n_time), ("temporal prior", temporal_prior)]

        if interaction is not None:
            inter = interaction_structure(R, Rt, interaction)
            con = null_space_constraints(inter["structure"])
            payload.update(
                interaction=interaction,
                interaction_structure=inter["structure"],
                interaction_rank=inter["rank"],
                rank_deficiency=inter["rank_deficiency"],
                n_constraints=con["n_constraints"],
                constraint_matrix=con["A"])
            lines += [("interaction", "Type " + interaction),
                      ("rank deficiency", inter["rank_deficiency"]),
                      ("constraints required", con["n_constraints"])]
            payload["constraint_note"] = (
                "Type I needs none, being of full rank; every other type is "
                "rank deficient and without A delta = 0 the interaction is "
                "confounded with the main time effect"
                if interaction == "I" else
                f"{con['n_constraints']} constraints are required, one per "
                f"unit of rank deficiency; without them the interaction is "
                f"confounded with the main time effect")

    payload["fitting_note"] = (
        "this specifies and diagnoses the model; fitting requires MCMC or "
        "INLA, and convergence is sensitive to the hyperprior parameters")
    return RichResult(title="Bayesian hierarchical spatial model",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spbayr: Bayesian hierarchical disease-mapping models (Sec. 6.4) "
            "-- exchangeable/ICAR/LCAR priors, RW temporal, Knorr-Held types")
