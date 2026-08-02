# morie.fn -- function file (rootcoder007/morie)
"""Falsification / pre-trend test."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["causal_falsification_test"]


def causal_falsification_test(y_pre, treat, X_baseline=None):
    r"""Test for a treatment effect where none can exist.

    Regresses a **pre-treatment** outcome on treatment. Since treatment has
    not happened yet, any association is confounding, not effect -- so this is
    a test of the design rather than of the hypothesis.

    Failing is informative and passing is not. A significant pre-treatment
    difference is direct evidence that the groups differ in ways the design
    does not handle. A null result is weak evidence in the other direction:
    the test has whatever power the sample gives it, and absence of a
    detectable pre-trend is routinely reported as though it established
    parallel trends, which it does not. ``power_note`` records the minimum
    detectable effect so the null can be read in proportion to what the test
    could have found.

    When multiple pre-periods are available, feeding each in turn is stronger
    than any single test, because a genuine violation usually shows a
    *pattern* rather than one significant coefficient.

    Parameters
    ----------
    y_pre : array-like
        Pre-treatment outcome.
    treat : array-like
        Treatment indicator, 0/1.
    X_baseline : array-like, optional
        Baseline covariates to adjust for.

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``p_value``, ``passed``,
        ``min_detectable_effect``, ``power_note``.

    References
    ----------
    Rosenbaum, P. R. (2002). *Observational Studies* (2nd ed.). Springer.

    Examples
    --------
    Under the null the test is correctly calibrated -- it rejects at about
    the nominal 5%, no more. Checking the rate across seeds is a real test of
    the procedure, where a single draw only shows whether that draw happened
    to be balanced.

    >>> import numpy as np
    >>> rej = 0
    >>> for s in range(400):
    ...     g = np.random.default_rng(s)
    ...     tr = g.integers(0, 2, 400).astype(float)
    ...     rej += not causal_falsification_test(g.normal(size=400), tr)["passed"]
    >>> bool(0.02 < rej / 400 < 0.09)
    True

    A confounded comparison fails, which is the informative direction.

    >>> rng = np.random.default_rng(0)
    >>> tr = rng.integers(0, 2, 1000).astype(float)
    >>> y0 = rng.normal(size=1000)
    >>> bool(not causal_falsification_test(y0 + 0.8 * tr, tr)["passed"])
    True

    Passing is weak evidence, so the minimum detectable effect is reported
    alongside -- a null from an underpowered test says very little.

    >>> small = causal_falsification_test(y0[:20], tr[:20])
    >>> big = causal_falsification_test(y0, tr)
    >>> bool(small["min_detectable_effect"] > big["min_detectable_effect"])
    True

    >>> causal_falsification_test([1.0, 2.0], [1, 1])
    Traceback (most recent call last):
        ...
    ValueError: both treatment groups must be non-empty
    """
    from ._stats_core import norm

    y = np.atleast_1d(np.asarray(y_pre, dtype=float)).ravel()
    tr = np.atleast_1d(np.asarray(treat, dtype=float)).ravel()
    if y.size != tr.size:
        raise ValueError(f"y_pre has {y.size} entries but treat has {tr.size}")
    if not np.all((tr == 0) | (tr == 1)):
        raise ValueError("treat must be 0/1")
    if not (tr == 1).any() or not (tr == 0).any():
        raise ValueError("both treatment groups must be non-empty")

    cols = [np.ones(y.size), tr]
    if X_baseline is not None:
        Xb = np.atleast_2d(np.asarray(X_baseline, dtype=float))
        if Xb.shape[0] != y.size:
            Xb = Xb.T
        cols.append(Xb)
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = max(y.size - A.shape[1], 1)
    s2 = float(resid @ resid / dof)
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        se = float("nan")
    est = float(beta[1])
    z = est / se if se > 0 else np.nan
    p = float(2 * norm.sf(abs(z))) if se > 0 else float("nan")
    mde = float(2.8 * se) if se > 0 else float("nan")
    return RichResult(
        title="Falsification (pre-trend) test",
        summary_lines=[("estimate", est), ("se", se), ("p", p),
                       ("min detectable effect", mde)],
        warnings=["failing is informative, passing is not: a null here has "
                  "only the power the sample gives it and does not establish "
                  "parallel trends"],
        payload={
            "estimate": est, "se": se, "z": float(z), "p_value": p,
            "passed": bool(p > 0.05) if np.isfinite(p) else False,
            "min_detectable_effect": mde,
            "power_note": ("a null result rules out effects larger than about "
                           f"{mde:.3g}, and nothing smaller"),
            "n": int(y.size), "method": "causal_falsification_test",
        },
    )


def cheatsheet():
    return "causflnk: tests the DESIGN not the hypothesis; failing is informative, passing is not"
