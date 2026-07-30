# morie.fn -- function file (rootcoder007/morie)
"""BayesCpi: spike-and-slab genomic regression with pi estimated."""

import numpy as np

from ._richresult import RichResult

__all__ = ["bayes_c_pi", "bayes_cpi_prior"]


def bayes_c_pi(y, X, n_iter=2000, burn_in=500, pi_a=1.0, pi_b=1.0,
               seed=0, nu=4.0, s2=None):
    r"""Gibbs sampler for the BayesC:math:`\pi` model.

    .. math::
       \beta_j \sim \pi\,\delta_0 + (1-\pi)\,N(0, \sigma_b^2),
       \qquad \pi \sim \mathrm{Beta}(p_0, p_1)

    Estimating :math:`\pi` rather than fixing it is the whole
    difference from BayesC, and it matters because :math:`\pi` IS the
    genetic architecture: a trait controlled by a few large-effect loci
    and one controlled by thousands of small ones need different
    shrinkage, and fixing :math:`\pi` imposes an answer the data should
    supply. When :math:`\pi \to 0` the model collapses to GBLUP-like
    ridge regression, so BayesC:math:`\pi` nests that case rather than
    competing with it.

    The sampler alternates: draw each :math:`\delta_j` from its full
    conditional inclusion probability, draw the included effects, then
    draw :math:`\pi` from its Beta conjugate given the current count of
    zeros. That last step is what "with :math:`\pi` estimated" means.

    ``pip`` gives the posterior inclusion probability per marker, which
    is the output worth using -- a single point estimate of
    :math:`\beta_j` hides whether the marker was in the model at all.
    ``rhat_pi`` splits the retained chain in half and compares
    variances; far from 1 means the chain has not mixed and the
    posterior summaries are not yet meaningful.

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, m)
        Centred marker matrix.
    n_iter, burn_in : int
    pi_a, pi_b : float
        Beta prior on the proportion of zero effects.
    seed : int
    nu, s2 : float
        Scaled inverse chi-square prior on the variances.

    Returns
    -------
    RichResult
        ``beta``, ``pip``, ``pi``, ``pi_samples``, ``sigma2_b``,
        ``sigma2_e``, ``n_selected``, ``rhat_pi``, ``converged``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), section 6.5,
    pp. 180-184.
    Habier, Fernando, Kizilkaya and Garrick (2011), *BMC
    Bioinformatics* 12:186.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 20))
    >>> y = X[:, 0] * 2.0 + rng.normal(scale=0.3, size=120)
    >>> out = bayes_c_pi(y, X, n_iter=400, burn_in=100)
    >>> int(np.argmax(out["pip"]))
    0
    """
    yv = np.asarray(y, dtype=float).ravel()
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    n = yv.size
    if Xa.shape[0] != n:
        Xa = Xa.T
    if Xa.shape[0] != n:
        raise ValueError("X has %d rows for %d phenotypes."
                         % (Xa.shape[0], n))
    m = Xa.shape[1]
    n_iter, burn_in = int(n_iter), int(burn_in)
    if burn_in >= n_iter:
        raise ValueError("burn_in must be smaller than n_iter.")
    if pi_a <= 0 or pi_b <= 0:
        raise ValueError("Beta prior parameters must be positive.")

    rng = np.random.default_rng(int(seed))
    yc = yv - yv.mean()
    xtx = np.sum(Xa ** 2, axis=0)
    xtx = np.where(xtx > 0, xtx, 1e-12)
    vy = float(np.var(yc, ddof=1)) or 1.0
    s2 = 0.5 * vy if s2 is None else float(s2)

    beta = np.zeros(m)
    delta = np.zeros(m, dtype=bool)
    s2e, s2b = 0.5 * vy, 0.5 * vy / max(m, 1)
    pi = 0.5
    resid = yc.copy()

    keep = n_iter - burn_in
    b_sum = np.zeros(m)
    d_sum = np.zeros(m)
    pis = np.empty(keep)
    s2e_s = np.empty(keep)
    s2b_s = np.empty(keep)
    ki = 0
    for it in range(n_iter):
        for j in range(m):
            if delta[j]:
                resid += Xa[:, j] * beta[j]
            rhs = float(Xa[:, j] @ resid)
            v = xtx[j] + s2e / s2b
            bhat = rhs / v
            # log odds of inclusion, on the log scale for stability
            log_bf = (0.5 * np.log(s2e / (s2b * v))
                      + 0.5 * rhs ** 2 / (s2e * v))
            odds = (1.0 - pi) / max(pi, 1e-12) * np.exp(
                np.clip(log_bf, -700, 700)
            )
            prob = odds / (1.0 + odds)
            if rng.uniform() < prob:
                delta[j] = True
                beta[j] = bhat + rng.normal() * np.sqrt(s2e / v)
                resid -= Xa[:, j] * beta[j]
            else:
                delta[j] = False
                beta[j] = 0.0
        k = int(delta.sum())
        # pi is DRAWN, not fixed: this is the "pi" in BayesCpi
        pi = float(rng.beta(pi_a + (m - k), pi_b + k))
        sse = float(resid @ resid)
        s2e = (sse + nu * s2) / rng.chisquare(n + nu)
        if k > 0:
            ssb = float(beta[delta] @ beta[delta])
            s2b = (ssb + nu * s2) / rng.chisquare(k + nu)
        if it >= burn_in:
            b_sum += beta
            d_sum += delta
            pis[ki] = pi
            s2e_s[ki] = s2e
            s2b_s[ki] = s2b
            ki += 1

    half = ki // 2
    rhat = np.nan
    if half > 1:
        v1, v2 = np.var(pis[:half], ddof=1), np.var(pis[half:ki], ddof=1)
        mu1, mu2 = pis[:half].mean(), pis[half:ki].mean()
        W = 0.5 * (v1 + v2)
        B = half * ((mu1 - mu2) ** 2) / 2.0
        rhat = float(np.sqrt(((half - 1) / half * W + B / half) / W)) \
            if W > 0 else np.nan
    pip = d_sum / max(ki, 1)
    return RichResult(
        payload={
            "estimate": b_sum / max(ki, 1),
            "beta": b_sum / max(ki, 1),
            "pip": pip,
            "pi": float(pis[:ki].mean()) if ki else np.nan,
            "pi_samples": pis[:ki],
            "pi_note": (
                "pi is drawn from its Beta conjugate each sweep rather than "
                "fixed; it IS the genetic architecture, and fixing it "
                "imposes an answer the data should give"
            ),
            "sigma2_b": float(s2b_s[:ki].mean()) if ki else np.nan,
            "sigma2_e": float(s2e_s[:ki].mean()) if ki else np.nan,
            "heritability": (float(np.mean(s2b_s[:ki] * m
                                           / (s2b_s[:ki] * m + s2e_s[:ki])))
                             if ki else np.nan),
            "n_selected": float(pip.sum()),
            "top_markers": np.argsort(pip)[::-1][:10],
            "pip_note": (
                "posterior inclusion probability is the output to use; a "
                "point estimate of beta_j hides whether the marker was in "
                "the model at all"
            ),
            "rhat_pi": rhat,
            "converged": bool(np.isfinite(rhat) and abs(rhat - 1.0) < 0.1),
            "rhat_note": (
                "split-half variance ratio for pi; far from 1 means the "
                "chain has not mixed and the summaries are not yet meaningful"
            ),
            "n_iter": n_iter,
            "burn_in": burn_in,
            "n_markers": int(m),
            "n": int(n),
            "method": "BayesCpi Gibbs sampler",
        }
    )


def cheatsheet():
    return (
        "byscn: BayesCpi spike-and-slab with pi drawn each sweep, returning "
        "posterior inclusion probabilities and a mixing check"
    )


#: Catalogue alias for :func:`bayes_c_pi`.
bayes_cpi_prior = bayes_c_pi
