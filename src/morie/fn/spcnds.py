# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conditional simulation of a Gaussian random field, by kriging."""

from . import _array_core as np

from ._richresult import RichResult
from ._schab_sim import simple_kriging_variance, simulate_conditional

__all__ = ["schabenberger_conditional_sim"]


def schabenberger_conditional_sim(cov_all, z_obs, n_obs, mean=0.0,
                                  method="cholesky", seed=0, stream=0):
    """Condition an unconditional simulation on observed values, Sec. 7.2.2.

    A conditional simulation "honors the observed values". The construction
    is eq (7.1): start from an unconditional simulation S with the same
    covariance, then correct it by the residual between the data and the
    simulation at the sampled locations,

        Zc(s) = psk(s; Z) + S(s) - psk(s; Sm)
              = S(s) + c' Sigma^-1 (Z - Sm).

    The text notes S need not carry the same mean -- "Any mean will do, for
    example, E[S(s)] = 0" -- so the unconditional draw here is centred and
    the mean rides in through the correction.

    Sec. 7.2.2 states three properties, all asserted in the test suites:
    the realization honors the data at the sampled locations, it is
    unconditionally unbiased, and it reproduces Cov[Z(s), Z(s+h)]. It also
    gives E[(Zc(s) - Z(s))^2] = 2 sigma^2_sk: a conditional simulation is
    deliberately twice as far from the truth in mean square as the kriging
    predictor, because "the idea of a conditional simulation is to reproduce
    data where it is known but not to smooth the data in-between".

    Parameters
    ----------
    cov_all : array-like, shape (n, n)
        Covariance over the observed locations followed by the targets.
    z_obs : array-like, shape (n_obs,)
        Observed values.
    n_obs : int
        How many leading rows of ``cov_all`` are observed locations.
    mean : float or array-like
        Mean of Z; scalar or one value per location.
    method : {"cholesky", "spectral"}
        Square root used for the unconditional draw (Sec. 7.1.1 or 7.1.2).
    seed, stream : int
        Generator handles.

    Returns
    -------
    RichResult
        Keys: ``field``, ``observed``, ``n_obs``, ``kriging_variance``,
        ``honors_data``, ``method``.

    References
    ----------
    Schabenberger Ch 7, Sec 7.2.2
    """
    cov_all = np.atleast_2d(np.asarray(cov_all, dtype=float))
    z_obs = np.asarray(z_obs, dtype=float).ravel()
    field = simulate_conditional(cov_all, z_obs, n_obs, mean=mean,
                                 method=method, seed=seed, stream=stream)
    n_obs = int(n_obs)
    honors = float(np.max(np.abs(field[:n_obs] - z_obs)))
    sk = simple_kriging_variance(cov_all, n_obs)
    return RichResult(
        title="Conditional simulation of a Gaussian random field",
        summary_lines=[("n", cov_all.shape[0]), ("observed", n_obs),
                       ("max departure at data", honors)],
        payload={"field": field, "observed": z_obs, "n_obs": n_obs,
                 "kriging_variance": sk, "honors_data": honors,
                 "n": int(cov_all.shape[0]), "method": method},
    )


def cheatsheet():
    return "spcnds: conditional simulation by kriging (Schabenberger Sec 7.2.2)"
