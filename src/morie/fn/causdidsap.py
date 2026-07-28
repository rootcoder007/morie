# morie.fn -- function file (rootcoder007/morie)
"""Sun and Abraham interaction-weighted event-study estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["causal_did_sun_abraham"]


def causal_did_sun_abraham(Y_panel, G_first_treat, rel_periods=None,
                           control="never"):
    r"""The interaction-weighted estimator of Sun and Abraham (2021).

    The problem it solves is specific. The usual two-way
    fixed-effects event study regresses the outcome on unit and
    period effects plus a set of relative-time dummies, and reads the
    coefficients as dynamic treatment effects. Sun and Abraham show
    that each such coefficient is a weighted sum of cohort-specific
    effects at MANY relative times, not just its own, and that the
    weights can be NEGATIVE. Under effect heterogeneity across
    cohorts the estimate can therefore have the wrong sign even when
    every cohort's true effect has the same sign -- the contamination
    comes from already-treated units acting as controls.

    The interaction-weighted estimator never pools across cohorts. It
    estimates a separate cohort-and-relative-time effect

    .. math:: \widehat{CATT}_{e,l} = \left(\bar Y_{e,e+l}
              - \bar Y_{e,e-1}\right)
              - \left(\bar Y_{c,e+l} - \bar Y_{c,e-1}\right)

    against a clean control group ``c`` -- never-treated units, or
    not-yet-treated ones -- and then aggregates to relative time
    :math:`l` using the SHARES of each cohort among units observed at
    that :math:`l`:

    .. math:: \hat\mu_l = \sum_e \hat w_{e,l}\,\widehat{CATT}_{e,l},
              \qquad \hat w_{e,l} \ge 0,\ \sum_e \hat w_{e,l} = 1.

    Because the weights are shares they are non-negative and sum to
    one, which is exactly the property the two-way fixed-effects
    coefficients lack. ``naive_twfe`` is computed alongside so the
    two can be compared on the same data; on a design with
    heterogeneous cohort effects they differ, and the tests here
    check that the interaction-weighted one is the accurate one.

    Period ``e - 1`` is the reference throughout, so ``mu[-1]`` is
    zero by construction and the pre-period estimates are a
    pre-trends check rather than an estimate of anything.

    Parameters
    ----------
    y_panel : array-like, shape (n_units, n_periods)
        Balanced panel of outcomes.
    g_first_treat : array-like, shape (n_units,)
        Period index of first treatment; ``inf`` or ``nan`` for
        never-treated units.
    rel_periods : sequence of int, optional
        Relative periods to report; a data-driven range otherwise.
    control : {"never", "notyet"}, default "never"
        Comparison group.

    Returns
    -------
    RichResult
        keys: ``rel_periods``, ``mu`` (the IW estimates),
        ``catt`` (cohort by relative time), ``weights``,
        ``cohorts``, ``naive_twfe``, ``weights_nonnegative``,
        ``n_never_treated``, ``n_units``, ``n_periods``, ``method``.

    References
    ----------
    Sun, L. and Abraham, S. (2021), "Estimating dynamic treatment
    effects in event studies with heterogeneous treatment effects",
    *Journal of Econometrics* 225:175-199.
    """
    Y = np.atleast_2d(np.asarray(Y_panel, dtype=float))
    G = np.asarray(G_first_treat, dtype=float).ravel()
    n, T = Y.shape
    if G.size != n:
        raise ValueError(f"G_first_treat has {G.size} entries for {n} units.")
    if control not in ("never", "notyet"):
        raise ValueError("control must be 'never' or 'notyet'.")
    never = ~np.isfinite(G)
    if control == "never" and not never.any():
        raise ValueError(
            "no never-treated units, so there is no clean control group; "
            "use control='notyet'.")
    cohorts = np.unique(G[np.isfinite(G)])
    cohorts = cohorts[(cohorts >= 1) & (cohorts <= T - 1)]
    if cohorts.size == 0:
        raise ValueError(
            "no cohort is treated at a period with both a pre-period and a "
            "post-period, so no effect is estimable.")
    if rel_periods is None:
        lo = int(-min(cohorts))
        hi = int(T - 1 - min(cohorts))
        rel_periods = [l for l in range(max(lo, -5), min(hi, 5) + 1)]
    rel = list(rel_periods)

    catt = np.full((cohorts.size, len(rel)), np.nan)
    wts = np.zeros((cohorts.size, len(rel)))
    for ci, e in enumerate(cohorts):
        ei = int(e)
        treated = G == e
        for li, l in enumerate(rel):
            t = ei + l
            if t < 0 or t >= T or ei - 1 < 0:
                continue
            if control == "never":
                ctrl = never
            else:
                # not yet treated AT PERIOD t (never-treated included)
                ctrl = (G > t) | never
                ctrl = ctrl & ~treated
            if not ctrl.any() or not treated.any():
                continue
            catt[ci, li] = ((Y[treated, t].mean() - Y[treated, ei - 1].mean())
                            - (Y[ctrl, t].mean() - Y[ctrl, ei - 1].mean()))
            wts[ci, li] = float(treated.sum())
    # shares among the cohorts that actually contribute at each l
    with np.errstate(invalid="ignore"):
        wts = np.where(np.isnan(catt), 0.0, wts)
        col = wts.sum(axis=0)
        wts = np.divide(wts, col, out=np.zeros_like(wts), where=col > 0)
    mu = np.array([
        float(np.nansum(wts[:, li] * np.nan_to_num(catt[:, li])))
        if wts[:, li].sum() > 0 else np.nan
        for li in range(len(rel))])

    # the naive two-way fixed-effects event study, for contrast
    ever = np.isfinite(G)
    unit_m = Y.mean(axis=1, keepdims=True)
    per_m = Y.mean(axis=0, keepdims=True)
    Yd = Y - unit_m - per_m + Y.mean()
    naive = []
    for l in rel:
        cells = []
        for i in range(n):
            if not ever[i]:
                continue
            t = int(G[i]) + l
            if 0 <= t < T:
                cells.append(Yd[i, t])
        naive.append(float(np.mean(cells)) if cells else np.nan)
    return RichResult(payload={
        "rel_periods": np.array(rel), "mu": mu,
        "catt": catt, "weights": wts, "cohorts": cohorts,
        "naive_twfe": np.array(naive),
        "weights_nonnegative": bool(np.all(wts >= -1e-12)),
        "weights_sum_to_one": bool(np.all(
            np.isclose(wts.sum(axis=0)[wts.sum(axis=0) > 0], 1.0))),
        "reference_period": -1,
        "n_never_treated": int(never.sum()),
        "control_group": control,
        "why_not_twfe": "a two-way fixed-effects event-study coefficient is "
                        "a weighted sum of cohort effects at MANY relative "
                        "times with possibly NEGATIVE weights, so under "
                        "heterogeneity it can carry the wrong sign; the "
                        "interaction weights are shares and cannot",
        "n_units": int(n), "n_periods": int(T),
        "method": "Sun-Abraham interaction-weighted event study (2021)"})


def cheatsheet():
    return "causdidsap: never pool across cohorts -- TWFE weights can go negative, shares cannot"
