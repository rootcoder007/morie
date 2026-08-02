# morie.fn -- function file (rootcoder007/morie)
"""Augmented inverse-probability-weighted estimator."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_aipw"]


def causal_aipw(y, T, ps, m1, m0, trim=0.01):
    r"""The augmented inverse-probability-weighted (doubly robust)
    estimator of the average treatment effect,

    .. math:: \hat\tau = \frac1n\sum_i\Big[
              \hat m_1(X_i) - \hat m_0(X_i)
              + \frac{T_i\{Y_i - \hat m_1(X_i)\}}{\hat e(X_i)}
              - \frac{(1-T_i)\{Y_i - \hat m_0(X_i)\}}{1 - \hat e(X_i)}
              \Big].

    The DOUBLE ROBUSTNESS is the property worth having and the one
    the tests exercise directly: the estimator is consistent if
    EITHER the propensity score :math:`\hat e` or the outcome
    regressions :math:`\hat m_1, \hat m_0` are correctly specified,
    not necessarily both. Feed it a wrong propensity model with right
    outcome models and it still lands on the truth; feed it a wrong
    outcome model with a right propensity model and it still lands on
    the truth; get both wrong and it does not, and no amount of
    doubly robust machinery rescues that.

    The reason it works is visible in the algebra. If :math:`\hat m`
    is right, the two augmentation terms have mean zero and the
    estimator is the regression estimator. If :math:`\hat e` is
    right, the terms rearrange into inverse-probability weighting
    with the regression as a variance-reducing control.

    Propensities near 0 or 1 are the practical failure mode: the
    estimator divides by them, so a handful of extreme scores can
    dominate the average. ``trim`` bounds them away from the
    boundary and ``n_trimmed`` says how many were touched, because
    trimming silently changes the estimand -- it is an average over
    the retained region, not over the whole population.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed outcome.
    t : array-like of 0/1, shape (n,)
        Treatment.
    ps : array-like, shape (n,)
        Estimated propensity score.
    m1, m0 : array-like, shape (n,)
        Estimated outcome regressions under treatment and control,
        evaluated for every unit.
    trim : float, default 0.01
        Propensities are clipped to ``[trim, 1 - trim]``.

    Returns
    -------
    RichResult
        keys: ``ate``, ``se``, ``ci``, ``influence``,
        ``regression_component``, ``augmentation_component``,
        ``n_trimmed``, ``min_ps``, ``max_ps``, ``effective_overlap``,
        ``doubly_robust``, ``n``, ``method``.

    References
    ----------
    Robins, J. M., Rotnitzky, A. and Zhao, L. P. (1994), "Estimation
    of regression coefficients when some regressors are not always
    observed", *JASA* 89:846-866. Bang and Robins (2005),
    *Biometrics* 61:962-973.
    """
    yv = np.asarray(y, dtype=float).ravel()
    Tv = np.asarray(T, dtype=float).ravel()
    e = np.asarray(ps, dtype=float).ravel()
    M1 = np.asarray(m1, dtype=float).ravel()
    M0 = np.asarray(m0, dtype=float).ravel()
    n = yv.size
    if not (Tv.size == e.size == M1.size == M0.size == n):
        raise ValueError("y, T, ps, m1 and m0 must have the same length.")
    if not np.all(np.isin(Tv, (0.0, 1.0))):
        raise ValueError("T must be binary 0/1.")
    if np.any(e < 0) or np.any(e > 1):
        raise ValueError("propensity scores must lie in [0, 1].")
    tr = float(trim)
    if not 0 <= tr < 0.5:
        raise ValueError(f"trim must lie in [0, 0.5), got {tr}.")
    n_trim = int(np.sum((e < tr) | (e > 1 - tr)))
    ec = np.clip(e, tr, 1 - tr) if tr > 0 else e
    if np.any(ec <= 0) or np.any(ec >= 1):
        raise ValueError(
            "a propensity score is exactly 0 or 1, so a unit has no "
            "counterfactual and the estimator divides by zero; trim > 0 or "
            "drop the unit.")
    reg = M1 - M0
    aug = Tv * (yv - M1) / ec - (1 - Tv) * (yv - M0) / (1 - ec)
    infl = reg + aug
    ate = float(infl.mean())
    se = float(np.std(infl, ddof=1) / np.sqrt(n))
    return RichResult(payload={
        "ate": ate, "se": se,
        "ci": (ate - 1.959963984540054 * se, ate + 1.959963984540054 * se),
        "influence": infl,
        "regression_component": float(reg.mean()),
        "augmentation_component": float(aug.mean()),
        "n_trimmed": n_trim, "min_ps": float(e.min()), "max_ps": float(e.max()),
        "effective_overlap": float(np.mean((e > 0.1) & (e < 0.9))),
        "doubly_robust": "consistent if EITHER the propensity score or the "
                         "outcome regressions are correct, not necessarily "
                         "both; wrong on both and it is wrong",
        "trimming_note": "trimming changes the estimand to an average over "
                         "the retained region, not the whole population",
        "n": int(n),
        "method": "AIPW / doubly robust ATE (Robins, Rotnitzky and Zhao 1994)"})


def cheatsheet():
    return "causaipw: right propensity OR right outcome model suffices -- both wrong and it fails"
