# morie.fn -- function file (rootcoder007/morie)
"""Maximum a posteriori ability estimation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["map_theta_estimator"]


def map_theta_estimator(y, a=None, b=None, c=None, prior=(0.0, 1.0),
                        bounds=(-6.0, 6.0)):
    r"""Maximum a posteriori (Bayes modal) estimate of :math:`\theta`
    -- the mode of

    .. math:: \log p(\theta \mid y) \propto \ell(\theta)
              - \frac{(\theta - \mu)^2}{2\sigma^2} .

    The prior's job here is precise: it makes the posterior
    log-concave in the tails regardless of the response pattern, so
    a MAP estimate EXISTS for every pattern including all-correct
    and all-wrong, where the maximum-likelihood estimate does not.
    That is why MAP is the usual production choice, and the tests
    check it on exactly those patterns.

    The price is SHRINKAGE toward the prior mean, which is largest
    where information is smallest -- short tests and extreme
    patterns. Shrinkage is not a bug but it is a bias, so
    ``shrinkage_vs_ml`` reports the gap from the maximum-likelihood
    estimate whenever the latter is finite, rather than leaving the
    user to assume there is none.

    The standard error is the posterior curvature
    :math:`1/\sqrt{I(\hat\theta) + 1/\sigma^2}` -- the prior
    contributes information, which is exactly why the MAP interval
    is narrower than the ML one.

    Parameters
    ----------
    y : array-like of 0/1
        Item responses.
    a, b, c : array-like, optional
        Item parameters; ``b`` required.
    prior : (float, float)
        Normal prior mean and standard deviation.
    bounds : (float, float)
        Search interval.

    Returns
    -------
    RichResult
        keys: ``theta``, ``se``, ``prior_mean``, ``prior_sd``,
        ``information``, ``posterior_information``,
        ``shrinkage_vs_ml``, ``exists_for_perfect_patterns`` (True),
        ``n_items``, ``method``.

    References
    ----------
    Samejima, F. (1969), "Estimation of latent ability using a
    response pattern of graded scores", *Psychometrika Monograph
    Supplement* 17. Bock, R. D. and Aitkin, M. (1981),
    *Psychometrika* 46:443-459. Mislevy, R. J. (1986),
    *Psychometrika* 51:177-195.
    """
    from ._psycho import logistic_3pl, logistic_3pl_deriv

    yv = np.asarray(y, dtype=float).ravel()
    m = yv.size
    if not np.all(np.isin(yv, (0.0, 1.0))):
        raise ValueError("responses must be binary 0/1.")
    if b is None:
        raise ValueError("item difficulties b are required.")
    bv = np.asarray(b, dtype=float).ravel()
    av = np.ones(m) if a is None else np.asarray(a, dtype=float).ravel()
    cv = np.zeros(m) if c is None else np.asarray(c, dtype=float).ravel()
    if not (bv.size == av.size == cv.size == m):
        raise ValueError("a, b, c must each have one entry per item.")
    mu, sd = float(prior[0]), float(prior[1])
    if sd <= 0:
        raise ValueError(f"the prior standard deviation must be positive, "
                         f"got {sd}.")
    grid = np.linspace(float(bounds[0]), float(bounds[1]), 8001)
    P = np.clip(logistic_3pl(grid, av, bv, cv), 1e-12, 1 - 1e-12)
    ll = (yv * np.log(P) + (1 - yv) * np.log(1 - P)).sum(axis=1)
    post = ll - (grid - mu) ** 2 / (2 * sd ** 2)
    th = float(grid[int(np.argmax(post))])
    # quadratic refinement on the winning triple
    i = int(np.argmax(post))
    if 0 < i < grid.size - 1:
        y0, y1, y2 = post[i - 1], post[i], post[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            th = float(grid[i] - 0.5 * (grid[1] - grid[0])
                       * (y2 - y0) / denom)
    Pt = np.clip(logistic_3pl(np.array([th]), av, bv, cv)[0],
                 1e-12, 1 - 1e-12)
    dP = logistic_3pl_deriv(np.array([th]), av, bv, cv)[0]
    info = float(np.sum(dP ** 2 / (Pt * (1 - Pt))))
    post_info = info + 1.0 / sd ** 2
    shrink = None
    if not (np.all(yv == 1) or np.all(yv == 0)):
        from .mleth import mle_theta_estimator
        ml = mle_theta_estimator(y, a=av, b=bv, c=cv, bounds=bounds)
        if ml["finite"]:
            shrink = float(th - ml["theta"])
    return RichResult(payload={
        "theta": th, "se": float(1 / np.sqrt(post_info)),
        "prior_mean": mu, "prior_sd": sd,
        "information": info, "posterior_information": post_info,
        "shrinkage_vs_ml": shrink,
        "exists_for_perfect_patterns": True,
        "why_it_exists": "the normal prior makes the posterior log-concave "
                         "in the tails whatever the pattern, so a mode "
                         "exists where the likelihood's maximum does not",
        "shrinkage_note": "the price is bias toward the prior mean, largest "
                          "where information is smallest -- short tests and "
                          "extreme patterns",
        "n_items": int(m),
        "method": "MAP (Bayes modal) theta under a normal prior"})


def cheatsheet():
    return "mapth: the prior buys existence for perfect patterns and pays in shrinkage -- both reported"
