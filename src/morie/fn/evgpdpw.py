# morie.fn -- function file (rootcoder007/morie)
"""PWM estimator of the generalised Pareto distribution."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["ev_gpd_pwm", "evt_gpd_pwm"]


def ev_gpd_pwm(x, threshold=None):
    r"""Hosking and Wallis (1987): fit the GPD to threshold excesses
    by probability-weighted moments,

    .. math:: \hat k = \frac{l_1}{l_2} - 2, \qquad
              \hat\sigma = l_1(1 + \hat k),

    on the excesses' first two L-moments. Hosking's :math:`k` is
    minus the GPD shape :math:`\xi`.

    Their Sec. 4 comparison is the reason to reach for this over
    maximum likelihood in practice: PWM has lower bias and comparable
    variance for :math:`-0.5 < k < 0.5`, and ML for the GPD often
    fails to converge at the sample sizes exceedances provide. The
    estimator is reliable only for :math:`\hat k > -0.5`
    (:math:`\xi < 0.5`, finite variance); outside it the output
    carries a warning rather than a silently untrustworthy number.

    Parameters
    ----------
    x : array-like
        Raw sample, or excesses when ``threshold`` is None.
    threshold : float, optional
        When given, excesses ``x[x > threshold] - threshold`` are
        formed here and the exceedance count reported.

    Returns
    -------
    RichResult
        keys: ``sigma``, ``k_hosking``, ``xi``, ``n_excesses``,
        ``threshold``, ``mean_excess``, ``reliable``,
        ``return_level_fn``, ``method``.

    References
    ----------
    Hosking, J. R. M. and Wallis, J. R. (1987), "Parameter and
    quantile estimation for the generalized Pareto distribution",
    *Technometrics* 29:339-349, Secs. 3-4.
    """
    from ._evt import gpd_from_pwm

    xv = np.asarray(x, dtype=float).ravel()
    if threshold is not None:
        u = float(threshold)
        exc = xv[xv > u] - u
    else:
        u = None
        exc = xv
        if np.any(exc < 0):
            raise ValueError("excesses must be non-negative; pass the "
                             "threshold to have them formed here.")
    n = exc.size
    if n < 10:
        raise ValueError(f"need at least 10 excesses, got {n}.")
    sigma, k = gpd_from_pwm(exc)
    xi = -k
    reliable = k > -0.5

    def return_level(m):
        """Level exceeded once every m excesses on average."""
        m = np.asarray(m, dtype=float)
        base = u if u is not None else 0.0
        if abs(k) < 1e-9:
            return base + sigma * np.log(m)
        return base + sigma / k * (1.0 - m ** (-k))

    return RichResult(payload={
        "sigma": sigma, "k_hosking": k, "xi": xi,
        "n_excesses": int(n), "threshold": u,
        "mean_excess": float(exc.mean()),
        "reliable": bool(reliable),
        "reliability_note": None if reliable else (
            "k <= -0.5 (xi >= 0.5): infinite variance territory, where the "
            "PWM estimator's own theory stops -- treat the numbers as "
            "indicative only"),
        "why_pwm": "lower bias than ML for -0.5 < k < 0.5 and none of ML's "
                   "convergence failures at exceedance sample sizes "
                   "(Hosking-Wallis Sec. 4)",
        "sign_convention": "Hosking's k = -xi",
        "return_level_fn": return_level,
        "method": "GPD by probability-weighted moments (Hosking-Wallis 1987)"})


def cheatsheet():
    return "evgpdpw: k = l1/l2 - 2, sigma = l1(1+k) -- and k <= -0.5 means the theory stopped"


#: Catalogue alias for :func:`ev_gpd_pwm`.
evt_gpd_pwm = ev_gpd_pwm
