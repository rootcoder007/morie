# morie.fn -- function file (rootcoder007/morie)
"""Bayesian Aldrich-McKelvey scaling via Gibbs sampling."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["bayesian_am_scaling"]


def bayesian_am_scaling(survey_data, n_iter=2000, burnin=500, seed=0, prior_sd=10.0):
    r"""Gibbs sampler for the BAM model.

    .. math:: z_{ik} = \alpha_i + \beta_i s_k + \varepsilon_{ik},
              \qquad \varepsilon \sim N(0, \sigma^2),

    with conjugate normal priors on each respondent's shift
    :math:`\alpha_i` and stretch :math:`\beta_i` (centred at 0 and 1)
    and on the stimulus positions s. Hare et al.'s Bayesian treatment
    handles missing placements and gives uncertainty for s directly;
    the classical estimator does not. Identification: s is
    re-normalised to mean 0 / sd 1 each sweep, so the posterior is
    over the identified scale.

    Parameters
    ----------
    survey_data : array-like, shape (n, q)
        Placements (NaN = missing).
    n_iter : int, default 2000
        Total sweeps.
    burnin : int, default 500
    seed : int, default 0
    prior_sd : float, default 10.0
        Prior sd for alpha and (beta - 1).

    Returns
    -------
    RichResult
        keys: ``stimuli`` (posterior mean, q,), ``stimuli_ci`` (2, q
        2.5/97.5%), ``alpha`` (n,), ``beta`` (n,), ``sigma``,
        ``n_kept``, ``n``, ``method``.

    References
    ----------
    Hare, C., Armstrong, D. A., Bakker, R., Carroll, R. & Poole, K. T.
    (2015). Using Bayesian Aldrich-McKelvey scaling to study
    citizens' ideological preferences and perceptions. *AJPS*, 59(3),
    759-774.

    Aldrich, J. H. & McKelvey, R. D. (1977). A method of scaling with
    applications to the 1968 and 1972 presidential elections. *APSR*,
    71(1), 111-130.
    """
    Z = np.asarray(survey_data, dtype=float)
    if Z.ndim != 2:
        raise ValueError("survey_data must be 2-D (respondents x stimuli).")
    n, q = Z.shape
    if q < 3:
        raise ValueError("need at least 3 stimuli.")
    n_iter, burnin = int(n_iter), int(burnin)
    if n_iter <= burnin:
        raise ValueError("n_iter must exceed burnin.")
    tau2 = float(prior_sd) ** 2
    if tau2 <= 0:
        raise ValueError("prior_sd must be positive.")

    obs = ~np.isnan(Z)
    if not obs.any():
        raise ValueError("all placements are missing.")
    rng = np.random.default_rng(seed)

    # initialise from column means
    s = np.nanmean(Z, axis=0)
    s = (s - s.mean()) / max(s.std(), 1e-8)
    alpha = np.zeros(n)
    beta = np.ones(n)
    sig2 = 1.0

    kept = []
    for it in range(n_iter):
        # respondent regressions z_i = alpha_i + beta_i s + e
        for i in range(n):
            m = obs[i]
            if m.sum() < 2:
                continue
            X = np.column_stack([np.ones(m.sum()), s[m]])
            prec = X.T @ X / sig2 + np.diag([1 / tau2, 1 / tau2])
            cov = np.linalg.inv(prec)
            mean = cov @ (X.T @ Z[i, m] / sig2 + np.array([0.0, 1.0 / tau2]))
            draw = rng.multivariate_normal(mean, cov)
            alpha[i], beta[i] = draw
        # stimulus positions given (alpha, beta)
        for k in range(q):
            m = obs[:, k]
            if not m.any():
                continue
            b = beta[m]
            prec = (b**2).sum() / sig2 + 1.0  # N(0,1) prior on s_k
            mean = (b * (Z[m, k] - alpha[m])).sum() / sig2 / prec
            s[k] = rng.normal(mean, np.sqrt(1 / prec))
        # identify: mean 0, sd 1 (absorb into alpha/beta)
        mu, sd = s.mean(), max(s.std(), 1e-8)
        s = (s - mu) / sd
        alpha = alpha + beta * mu
        beta = beta * sd
        # sigma^2 | rest, IG(1, 1) prior
        resid = Z[obs] - (alpha[:, None] + beta[:, None] * s[None, :])[obs]
        sh = 1.0 + obs.sum() / 2.0
        sc = 1.0 + (resid**2).sum() / 2.0
        sig2 = sc / rng.gamma(sh)
        if it >= burnin:
            kept.append(s.copy())

    draws = np.array(kept)
    return RichResult(
        payload={
            "stimuli": draws.mean(axis=0),
            "stimuli_ci": np.percentile(draws, [2.5, 97.5], axis=0),
            "alpha": alpha,
            "beta": beta,
            "sigma": float(np.sqrt(sig2)),
            "n_kept": int(draws.shape[0]),
            "n": int(n),
            "method": "Bayesian Aldrich-McKelvey (Gibbs; Hare et al. 2015)",
        }
    )


def cheatsheet():
    return "bayam: Gibbs over (alpha_i, beta_i), s, sigma; s renormalised each sweep"
