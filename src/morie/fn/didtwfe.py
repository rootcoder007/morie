# morie.fn -- function file (rootcoder007/morie)
"""Two-way fixed-effects difference-in-differences."""

from . import _array_core as np

from ._did import as_panel, cluster_se, first_treatment, twfe_beta
from ._richresult import RichResult

__all__ = ["twoway_fixed_effects_did"]


def twoway_fixed_effects_did(y, D, unit, time):
    r"""The two-way fixed-effects estimator, reported with its own diagnosis.

    The regression is

    .. math:: y_{it} = \alpha_i + \lambda_t + \beta D_{it}
              + \varepsilon_{it},

    and for a long time :math:`\hat\beta` was read as "the" DiD
    effect. Goodman-Bacon (2021) showed what it actually is: a
    weighted average of every 2x2 DiD available in the panel,
    including comparisons that use ALREADY-TREATED units as controls.
    Those comparisons difference out the earlier cohort's own
    treatment path, so under effect heterogeneity across cohorts they
    can enter with negative weight and :math:`\hat\beta` can have the
    wrong sign even when every unit's true effect is positive.

    This function therefore returns :math:`\hat\beta` **and** the
    facts needed to decide whether to believe it.
    ``already_treated_share`` is the fraction of the controls
    available to treated cells that are themselves already treated --
    the forbidden comparisons -- and ``timing_varies`` says whether
    the problem can arise at all. With a single adoption date the
    whole staggered-timing literature is moot and this estimator is
    the DiD estimator; the diagnosis says so rather than leaving the
    reader to infer it.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome in long format.
    D : array-like, shape (n,)
        Binary treatment indicator; must be absorbing.
    unit, time : array-like, shape (n,)
        Unit and period identifiers. The panel must be balanced.

    Returns
    -------
    RichResult
        ``estimate``, ``se`` (unit-clustered), ``t``, ``ci``,
        ``n_units``, ``n_periods``, ``cohorts``, ``timing_varies``,
        ``already_treated_share``, ``trustworthy``, ``diagnosis``.

    Notes
    -----
    The standard error clusters on the unit. Bertrand, Duflo and
    Mullainathan (2004) showed that ignoring serial correlation
    within a unit inflates DiD t-statistics severalfold, so an
    unclustered DiD standard error is not a defensible default.

    References
    ----------
    Goodman-Bacon (2021), *Journal of Econometrics* 225:254-277.
    Bertrand, Duflo and Mullainathan (2004), *QJE* 119:249-275.
    de Chaisemartin and D'Haultfoeuille (2020), *AER* 110:2964-2996.

    Examples
    --------
    >>> import numpy as np
    >>> unit = np.repeat(np.arange(6), 6)
    >>> time = np.tile(np.arange(6), 6)
    >>> g = np.repeat([2.0, 2.0, 4.0, 4.0, np.inf, np.inf], 6)
    >>> D = (time >= g).astype(float)
    >>> y = unit * 0.1 + time * 0.2 + 1.0 * D
    >>> round(twoway_fixed_effects_did(y, D, unit, time)["estimate"], 6)
    1.0
    """
    Y, units, periods = as_panel(y, unit, time)
    g, Dm, _, _ = first_treatment(D, unit, time, units, periods)
    beta, resid, Dt, denom = twfe_beta(Y, Dm)
    se = cluster_se(resid, Dt, denom, len(units))

    cohorts = np.unique(g[np.isfinite(g)])
    timing_varies = bool(cohorts.size > 1)
    # of the controls available to a treated cell, what share are
    # themselves already treated?
    treated = Dm > 0
    n_treated_t = treated.sum(axis=0).astype(float)
    active = n_treated_t > 0
    if active.any() and len(units) > 1:
        at_share = float(
            np.average(
                (n_treated_t[active] - 1.0) / (len(units) - 1.0),
                weights=n_treated_t[active],
            )
        )
    else:
        at_share = 0.0

    t = beta / se if se > 0 else np.nan
    trust = (not timing_varies) or at_share < 1e-12
    z = 1.959963984540054
    return RichResult(
        payload={
            "estimate": beta,
            "se": se,
            "t": float(t),
            "ci": (beta - z * se, beta + z * se),
            "n_units": int(len(units)),
            "n_periods": int(len(periods)),
            "n_treated_cells": int(treated.sum()),
            "cohorts": cohorts,
            "timing_varies": timing_varies,
            "already_treated_share": at_share,
            "trustworthy": bool(trust),
            "diagnosis": (
                "single adoption date: the staggered-timing critique does "
                "not apply and this coefficient is the DiD estimate"
                if not timing_varies
                else "treatment timing varies, so this coefficient is a "
                "weighted sum of 2x2 comparisons including already-treated "
                "controls; use goodman_bacon_decomp to see the weights and a "
                "heterogeneity-robust estimator to replace it"
            ),
            "se_note": "clustered on unit (Bertrand-Duflo-Mullainathan 2004)",
            "method": "Two-way fixed-effects DiD with Goodman-Bacon diagnosis",
        }
    )


def cheatsheet():
    return (
        "didtwfe: TWFE DiD coefficient plus the staggered-timing diagnosis "
        "(share of controls that are already treated)"
    )
