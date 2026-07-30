# morie.fn -- function file (rootcoder007/morie)
"""Propensity trimming for the trimmed causal effect."""

import numpy as np

from ._richresult import RichResult
from ._did import add_intercept, logit_fit, logit_predict

__all__ = ["trimmed_causal_effect", "truncated_cf_estimator"]


def trimmed_causal_effect(y, d, X=None, propensity=None, alpha=None,
                          rule="crump"):
    r"""Trim on the propensity, and say what estimand survives.

    Crump, Hotz, Imbens and Mitnik show that the variance-minimising
    subsample is :math:`\{\alpha \le e(X) \le 1-\alpha\}` with
    :math:`\alpha` solving

    .. math::
       \frac{2}{\alpha(1-\alpha)}
       = \mathbb{E}\!\left[\frac{1}{e(X)(1-e(X))}
         \;\middle|\;
         \frac{1}{e(X)(1-e(X))} \le \frac{2}{\alpha(1-\alpha)}\right],

    and that in practice :math:`\alpha = 0.1` is close to optimal
    across a wide range of designs -- which is where the familiar rule
    of thumb comes from, rather than from convention.

    Trimming is not a robustness fix, and the distinction is the point
    of this function. It CHANGES THE ESTIMAND: the answer becomes the
    effect in the trimmed subpopulation, which is defined by covariate
    values rather than by anything of substantive interest, and it may
    not be the population any decision concerns. ``estimand_note`` and
    ``n_dropped`` make the trade explicit rather than burying it in a
    preprocessing step.

    The overlap it buys is real. ``variance_reduction`` gives the ratio
    of the untrimmed to trimmed variance factor
    :math:`E[1/(e(1-e))]`, which is how much precision was gained, and
    ``dropped_covariate_shift`` reports how different the discarded
    units were -- if they differ sharply, the trimmed estimand is a
    long way from the original.

    Parameters
    ----------
    y : array-like, shape (n,)
    d : array-like of {0, 1}, shape (n,)
    X : array-like, optional
    propensity : array-like, optional
    alpha : float, optional
        Trimming bound; solved for when omitted under ``rule='crump'``.
    rule : {'crump', 'fixed'}

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``alpha``, ``n_kept``, ``n_dropped``,
        ``variance_reduction``, ``dropped_covariate_shift``,
        ``untrimmed_estimate``.

    References
    ----------
    Crump, Hotz, Imbens and Mitnik (2009), *Biometrika* 96:187-199.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 1))
    >>> d = (rng.uniform(size=600) < 1 / (1 + np.exp(-2 * X[:, 0]))).astype(float)
    >>> y = 2.0 * d + X[:, 0] + rng.normal(size=600)
    >>> out = trimmed_causal_effect(y, d, X)
    >>> bool(out["n_dropped"] >= 0)
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    n = yv.size
    if dv.size != n:
        raise ValueError("y and d must agree in length.")
    if not np.all(np.isin(dv, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")
    if rule not in ("crump", "fixed"):
        raise ValueError("rule must be 'crump' or 'fixed', got %r." % rule)
    Xa = None if X is None else np.atleast_2d(np.asarray(X, dtype=float))
    if Xa is not None and Xa.shape[0] != n:
        Xa = Xa.T
    if propensity is None:
        if Xa is None:
            raise ValueError("supply X or propensity.")
        B = add_intercept(Xa)
        e = logit_predict(B, logit_fit(B, dv)[0])
    else:
        e = np.asarray(propensity, dtype=float).ravel()
        if e.size != n:
            raise ValueError("propensity has %d entries for %d rows."
                             % (e.size, n))
    e = np.clip(e, 1e-6, 1 - 1e-6)
    inv = 1.0 / (e * (1 - e))

    if alpha is not None:
        a = float(alpha)
        if not 0 <= a < 0.5:
            raise ValueError("alpha must lie in [0, 0.5), got %r." % a)
    elif rule == "fixed":
        a = 0.1
    else:
        # Crump's condition is a CROSSING, not an inequality. Define
        # g(gamma) = 2 E[inv | inv <= gamma] - gamma. At small gamma the
        # retained values all sit near gamma so g > 0; as gamma grows the
        # conditional mean flattens to E[inv] and g goes negative. The
        # optimal cutoff is the crossing.
        #
        # Testing "mean(inv[inv <= c]) <= c" instead is satisfied at the
        # very first candidate -- the mean of values below c is always
        # below c -- so it returns gamma too small, the discriminant goes
        # negative and alpha silently comes back 0 with nothing trimmed.
        cands = np.unique(np.sort(inv))
        cands = cands[cands >= 4.0]
        gamma = None
        for cand in cands:
            m = inv <= cand
            if m.sum() < 4:
                continue
            if 2.0 * float(np.mean(inv[m])) - cand <= 0.0:
                gamma = float(cand)
                break
        if gamma is None or gamma <= 8.0:
            # no cutoff improves the variance: overlap is already good
            a = 0.0
        else:
            disc = 1.0 - 8.0 / gamma
            a = 0.5 * (1.0 - np.sqrt(max(disc, 0.0)))
        a = float(np.clip(a, 0.0, 0.4999))

    keep = (e >= a) & (e <= 1 - a)
    if keep.sum() < 4 or dv[keep].sum() < 2 or (1 - dv[keep]).sum() < 2:
        raise ValueError(
            "trimming at alpha = %.3f leaves too few units in one arm." % a
        )

    def ate(mask):
        ee = e[mask]
        dd, yy = dv[mask], yv[mask]
        w1 = dd / ee
        w0 = (1 - dd) / (1 - ee)
        return float((w1 @ yy) / w1.sum() - (w0 @ yy) / w0.sum())

    est = ate(keep)
    untrimmed = ate(np.ones(n, dtype=bool))
    psi = np.where(keep,
                   dv * yv / e - (1 - dv) * yv / (1 - e) - est, 0.0)
    se = float(np.std(psi[keep], ddof=1) / np.sqrt(keep.sum()))

    shift = np.nan
    if Xa is not None and (~keep).any() and keep.any():
        sd = Xa.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        shift = float(np.max(np.abs(
            (Xa[~keep].mean(axis=0) - Xa[keep].mean(axis=0)) / sd
        )))
    return RichResult(
        payload={
            "estimate": est,
            "se": se,
            "ci": (est - 1.959963984540054 * se,
                   est + 1.959963984540054 * se),
            "untrimmed_estimate": untrimmed,
            "alpha": float(a),
            "rule": rule,
            "n_kept": int(keep.sum()),
            "n_dropped": int((~keep).sum()),
            "kept": keep,
            "estimand_note": (
                "trimming CHANGES the estimand: this is the effect in the "
                "subpopulation with propensity between %.3f and %.3f, a "
                "group defined by covariate values rather than by anything "
                "of substantive interest, and possibly not the population "
                "any decision concerns" % (a, 1 - a)
            ),
            "variance_reduction": (float(np.mean(inv) / np.mean(inv[keep]))
                                   if keep.any() else np.nan),
            "variance_note": (
                "ratio of the untrimmed to trimmed E[1/(e(1-e))], which is "
                "the precision actually bought"
            ),
            "dropped_covariate_shift": shift,
            "shift_note": (
                "largest standardised covariate difference between dropped "
                "and kept units; a large value means the trimmed estimand is "
                "far from the original one"
            ),
            "propensity_range": (float(e.min()), float(e.max())),
            "n": int(n),
            "method": "Trimmed causal effect (%s rule)" % rule,
        }
    )


def cheatsheet():
    return (
        "trncfg: Crump-optimal propensity trimming, reporting the precision "
        "gained and the estimand given up"
    )


#: Catalogue alias for :func:`trimmed_causal_effect`.
truncated_cf_estimator = trimmed_causal_effect
