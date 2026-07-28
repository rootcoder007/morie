# morie.fn -- function file (rootcoder007/morie)
"""REML estimator of between-study variance."""

import numpy as np

from ._richresult import RichResult

__all__ = ["ma_random_reml"]


def ma_random_reml(yi, vi, max_iter=200, tol=1e-12):
    r"""Restricted maximum likelihood for the random-effects
    meta-analysis between-study variance, iterating Viechtbauer's
    (2005) update

    .. math:: \tau^2_{new} = \frac{\sum_i w_i^2\{(y_i-\hat\mu)^2
              - v_i\} + 1/\sum_i w_i}{\sum_i w_i^2},
              \qquad w_i = \frac1{v_i + \tau^2},

    truncated at zero.

    REML rather than ML for one reason and it is the same reason a
    sample variance divides by :math:`n-1`: ordinary maximum
    likelihood does not account for the degree of freedom spent
    estimating :math:`\mu`, and its :math:`\tau^2` is biased
    DOWNWARD as a result. The :math:`1/\sum w_i` term in the
    numerator is precisely that correction, and dropping it recovers
    the ML estimator -- which is returned alongside so the size of
    the correction is visible rather than assumed negligible.

    Convergence is not guaranteed (this is a fixed-point iteration on
    a possibly flat likelihood), so ``converged`` is reported and a
    failure is not disguised as an answer.

    Parameters
    ----------
    yi, vi : array-like
        Effects and their within-study variances.
    max_iter : int, default 200
        Iterations.
    tol : float, default 1e-12
        Convergence tolerance on tau^2.

    Returns
    -------
    RichResult
        keys: ``tau2``, ``tau``, ``mu``, ``se``, ``ci``, ``tau2_ml``,
        ``tau2_dl``, ``reml_correction``, ``converged``,
        ``n_iter``, ``I2``, ``k``, ``method``.

    References
    ----------
    Viechtbauer, W. (2005), "Bias and efficiency of meta-analytic
    variance estimators in the random-effects model", *Journal of
    Educational and Behavioral Statistics* 30:261-293.
    Viechtbauer, W. (2010), *Journal of Statistical Software*
    36(3), for the metafor implementation.
    """
    from scipy import stats

    from ._psycho import dersimonian_laird, fixed_effect_pool

    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    _, _, Q0, _ = fixed_effect_pool(y, v)
    k = y.size

    def iterate(reml):
        t2 = max(0.0, float(dersimonian_laird(y, v)))
        conv = False
        it = 0
        for it in range(1, int(max_iter) + 1):
            w = 1.0 / (v + t2)
            mu = float(np.sum(w * y) / np.sum(w))
            num = float(np.sum(w ** 2 * ((y - mu) ** 2 - v)))
            if reml:
                num += 1.0 / float(np.sum(w))
            new = max(0.0, num / float(np.sum(w ** 2)))
            if abs(new - t2) < tol * max(1.0, t2):
                t2 = new
                conv = True
                break
            t2 = new
        return t2, conv, it

    t2, conv, nit = iterate(True)
    t2_ml, _, _ = iterate(False)
    w = 1.0 / (v + t2)
    mu = float(np.sum(w * y) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    z = stats.norm.ppf(0.975)
    return RichResult(payload={
        "tau2": float(t2), "tau": float(np.sqrt(t2)),
        "mu": mu, "se": se, "ci": (mu - z * se, mu + z * se),
        "tau2_ml": float(t2_ml),
        "tau2_dl": float(dersimonian_laird(y, v)),
        "reml_correction": float(t2 - t2_ml),
        "why_reml": "ordinary ML does not account for the degree of freedom "
                    "spent estimating mu and is biased DOWNWARD; the "
                    "1/sum(w) term is exactly that correction, the same "
                    "reason a sample variance divides by n - 1",
        "converged": bool(conv), "n_iter": int(nit),
        "convergence_note": "a fixed-point iteration on a possibly flat "
                            "likelihood; a failure is reported, not "
                            "disguised",
        "Q": Q0,
        "I2": float(max(0.0, (Q0 - (k - 1)) / Q0)) if Q0 > 0 else 0.0,
        "k": int(k),
        "method": "REML tau^2 by fixed-point iteration (Viechtbauer 2005)"})


def cheatsheet():
    return "mareml: the 1/sum(w) term IS the REML correction -- drop it and you have biased ML"
