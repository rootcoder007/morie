# morie.fn -- function file (rootcoder007/morie)
"""Wald estimator for a binary instrument."""

import numpy as np

from ._richresult import RichResult

__all__ = ["wald_estimator"]


def wald_estimator(y, d, z, alpha=0.05):
    r"""Ratio of reduced form to first stage.

    .. math::
       \hat\beta_{Wald} =
       \frac{E[Y \mid Z=1] - E[Y \mid Z=0]}
            {E[D \mid Z=1] - E[D \mid Z=0]}

    With a binary instrument and one-sided noncompliance this is the
    LATE: the effect among COMPLIERS, those who take the treatment when
    encouraged and not otherwise. It is not the ATE, and the difference
    is not a technicality -- compliers are defined by their response to
    the instrument, and there is no way to identify who they are.

    The denominator is the whole story. It is the compliance rate, and
    as it shrinks the estimate becomes the ratio of two noisy
    quantities, whose distribution is heavy-tailed and whose Wald
    interval loses coverage. Conventional practice treats a first-stage
    F below 10 as weak; ``first_stage_f`` and ``weak_instrument`` apply
    that test rather than leaving it to the reader. With a genuinely
    weak instrument the point estimate is biased TOWARD the OLS
    estimate, which is the direction that makes it look reassuring.

    The monotonicity assumption -- no defiers -- is what rules out the
    denominator being a difference of two opposing groups that partly
    cancel. It is untestable, and violating it can put the estimate
    outside the range of any individual effect.

    Parameters
    ----------
    y : array-like, shape (n,)
    d : array-like of {0, 1}, shape (n,)
        Treatment actually taken.
    z : array-like of {0, 1}, shape (n,)
        Instrument.
    alpha : float

    Returns
    -------
    RichResult
        ``estimate``, ``se``, ``ci``, ``reduced_form``,
        ``first_stage``, ``first_stage_f``, ``weak_instrument``,
        ``complier_share``.

    References
    ----------
    Molak (2023), *Causal Inference and Discovery in Python*, Packt,
    chapter 6. Imbens and Angrist (1994), *Econometrica* 62:467-475.
    Staiger and Stock (1997) for the F > 10 convention.

    Examples
    --------
    >>> # one-sided noncompliance: nobody in the control arm is treated
    >>> y = [0, 0, 0, 0, 0, 1, 1, 1]
    >>> d = [0, 0, 0, 0, 0, 1, 1, 1]
    >>> z = [0, 0, 0, 0, 1, 1, 1, 1]
    >>> float(wald_estimator(y, d, z)["estimate"])
    1.0
    """
    yv = np.asarray(y, dtype=float).ravel()
    dv = np.asarray(d, dtype=float).ravel()
    zv = np.asarray(z, dtype=float).ravel()
    n = yv.size
    if not (dv.size == zv.size == n):
        raise ValueError(
            "y, d and z must agree in length, got %d, %d and %d."
            % (n, dv.size, zv.size)
        )
    if not np.all(np.isin(zv, (0.0, 1.0))):
        raise ValueError("z must be binary 0/1.")
    if not np.all(np.isin(dv, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1.")
    m1, m0 = zv == 1, zv == 0
    if m1.sum() < 2 or m0.sum() < 2:
        raise ValueError(
            "need at least 2 observations at each instrument value, got "
            "%d and %d." % (int(m1.sum()), int(m0.sum()))
        )
    rf = float(yv[m1].mean() - yv[m0].mean())
    fs = float(dv[m1].mean() - dv[m0].mean())
    if abs(fs) < 1e-12:
        raise ValueError(
            "the first stage is zero: the instrument does not move "
            "treatment, so the Wald ratio is undefined."
        )
    beta = rf / fs

    n1, n0 = int(m1.sum()), int(m0.sum())
    v_rf = yv[m1].var(ddof=1) / n1 + yv[m0].var(ddof=1) / n0
    v_fs = dv[m1].var(ddof=1) / n1 + dv[m0].var(ddof=1) / n0
    # Delta method on the ratio, INCLUDING the Cov(num, den) term. Y and
    # D are measured on the same subjects, so the numerator and
    # denominator are correlated and dropping the covariance -- which
    # most textbook presentations do -- biases the standard error. The
    # sign of the bias follows the sign of Cov(Y, D): with the usual
    # positive correlation the naive SE is too LARGE.
    c_yd = (np.cov(yv[m1], dv[m1], ddof=1)[0, 1] / n1
            + np.cov(yv[m0], dv[m0], ddof=1)[0, 1] / n0)
    se = float(np.sqrt(max(
        v_rf / fs ** 2
        + rf ** 2 * v_fs / fs ** 4
        - 2.0 * (rf / fs ** 3) * c_yd,
        0.0,
    )))
    f = float(fs ** 2 / v_fs) if v_fs > 0 else np.inf
    z95 = 1.959963984540054
    return RichResult(
        payload={
            "estimate": beta,
            "late": beta,
            "se": se,
            "ci": (beta - z95 * se, beta + z95 * se),
            "reduced_form": rf,
            "first_stage": fs,
            "cov_term": float(c_yd),
            "cov_note": (
                "Y and D are measured on the same subjects, so the "
                "reduced form and first stage are correlated; the "
                "delta method here keeps that covariance, which most "
                "textbook formulas drop"
            ),
            "first_stage_f": f,
            "weak_instrument": bool(f < 10.0),
            "weak_note": (
                None if f >= 10.0 else
                "first-stage F is %.1f, below the conventional 10; the Wald "
                "ratio is then heavy-tailed, its interval undercovers, and "
                "the bias runs TOWARD the OLS estimate" % f
            ),
            "complier_share": float(fs),
            "estimand_note": (
                "this is the LATE, the effect among compliers -- those who "
                "take treatment when encouraged and not otherwise. Compliers "
                "cannot be identified individually, and this is not the ATE"
            ),
            "monotonicity_note": (
                "identification needs no defiers; the assumption is "
                "untestable, and violating it can push the estimate outside "
                "the range of every individual effect"
            ),
            "n_encouraged": n1,
            "n_control": n0,
            "n": int(n),
            "method": "Wald estimator for a binary instrument",
        }
    )


def cheatsheet():
    return (
        "waldr: reduced form over first stage, with the weak-instrument F "
        "and the complier-only estimand stated"
    )
