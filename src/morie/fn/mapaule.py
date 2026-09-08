# morie.fn -- function file (rootcoder007/morie)
"""Paule-Mandel estimator of between-study variance."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ma_paule_mandel"]


def ma_paule_mandel(yi, vi, max_iter=200, tol=1e-12):
    r"""The Paule-Mandel (1982) estimator of the between-study
    variance: the :math:`\tau^2 \ge 0` solving

    .. math:: \sum_i \frac{(y_i - \hat\mu(\tau^2))^2}{v_i + \tau^2}
              = k - 1, \qquad
              \hat\mu(\tau^2) = \frac{\sum_i y_i/(v_i+\tau^2)}
                                     {\sum_i 1/(v_i+\tau^2)} .

    In words: choose :math:`\tau^2` so the generalised Q statistic
    equals its expectation. The left side is strictly decreasing in
    :math:`\tau^2`, so the root is unique and simple bisection finds
    it -- no convergence failure mode, unlike the likelihood-based
    estimators.

    Why it matters which estimator is used: the pooled estimate's
    weights are :math:`1/(v_i + \tau^2)`, so a different
    :math:`\tau^2` is a DIFFERENT pooled effect and a different
    confidence interval, not just a different heterogeneity
    footnote. Paule-Mandel is markedly less downward-biased than
    DerSimonian-Laird (Veroniki et al. 2016 review it), and the
    DL value is returned alongside so the gap is visible.

    When Q is already below its expectation at :math:`\tau^2 = 0`
    the root is at the boundary and :math:`\hat\tau^2 = 0` -- a
    truncation, not an estimate, and ``at_boundary`` says so.

    Parameters
    ----------
    yi : array-like
        Study effect estimates.
    vi : array-like
        Their within-study variances, positive.
    max_iter : int, default 200
        Bisection iterations.
    tol : float, default 1e-12
        Bracket tolerance.

    Returns
    -------
    RichResult
        keys: ``tau2``, ``tau``, ``mu``, ``se``, ``ci``, ``Q``,
        ``I2``, ``tau2_dl``, ``at_boundary``, ``weights``, ``k``,
        ``method``.

    References
    ----------
    Paule, R. C. and Mandel, J. (1982), "Consistency tests for the
    estimation of interlaboratory means", *Journal of Research of
    the NBS* 87:377-385. Veroniki et al. (2016), *Research Synthesis
    Methods* 7:55-79, for the comparison of tau^2 estimators.
    """
    from . import _stats_core as stats

    from ._psycho import dersimonian_laird, fixed_effect_pool

    y = np.asarray(yi, dtype=float).ravel()
    v = np.asarray(vi, dtype=float).ravel()
    _, _, Q0, _ = fixed_effect_pool(y, v)
    k = y.size

    def gen_q(t2):
        w = 1.0 / (v + t2)
        mu = float(np.sum(w * y) / np.sum(w))
        return float(np.sum(w * (y - mu) ** 2))

    at_boundary = gen_q(0.0) <= k - 1
    if at_boundary:
        t2 = 0.0
    else:
        lo, hi = 0.0, max(1.0, float(np.var(y, ddof=1)))
        while gen_q(hi) > k - 1:
            hi *= 2.0
            if hi > 1e12:
                break
        for _ in range(int(max_iter)):
            mid = 0.5 * (lo + hi)
            if gen_q(mid) > k - 1:
                lo = mid
            else:
                hi = mid
            if hi - lo < tol * max(1.0, hi):
                break
        t2 = 0.5 * (lo + hi)
    w = 1.0 / (v + t2)
    mu = float(np.sum(w * y) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    z = stats.norm.ppf(0.975)
    return RichResult(payload={
        "tau2": float(t2), "tau": float(np.sqrt(t2)),
        "mu": mu, "se": se, "ci": (mu - z * se, mu + z * se),
        "Q": Q0, "I2": float(max(0.0, (Q0 - (k - 1)) / Q0)) if Q0 > 0
        else 0.0,
        "tau2_dl": float(dersimonian_laird(y, v)),
        "at_boundary": bool(at_boundary),
        "boundary_note": None if not at_boundary else (
            "Q is already below its expectation at tau^2 = 0, so the "
            "estimate is a truncation at the boundary rather than an "
            "interior solution"),
        "weights": w, "k": int(k),
        "uniqueness_note": "the generalised Q is strictly decreasing in "
                           "tau^2, so the root is unique and bisection "
                           "cannot fail -- unlike the likelihood-based "
                           "estimators",
        "why_it_matters": "the weights are 1/(v_i + tau^2), so a different "
                          "tau^2 is a different POOLED EFFECT, not just a "
                          "heterogeneity footnote",
        "method": "Paule-Mandel (1982) tau^2 by bisection on the generalised Q"})


def cheatsheet():
    return "mapaule: solve generalised Q = k - 1 -- unique root, and tau^2 changes the pooled effect"


# compact alias per ledger/NAMING.md
mapaulemandel = ma_paule_mandel
