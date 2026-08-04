# morie.fn -- function file (rootcoder007/morie)
"""TMLE for the instrumental-variable LATE."""

from . import _array_core as np

from ._richresult import RichResult
from ._tmle import tmle_ate

__all__ = ["tmle_iv"]


def tmle_iv(y, D, Z, covariates=None, trunc=0.01):
    r"""LATE as a ratio of two targeted estimates.

    Under instrument validity and monotonicity,

    .. math:: \mathrm{LATE} = \frac{E[Y(z{=}1)] - E[Y(z{=}0)]}
              {E[D(z{=}1)] - E[D(z{=}0)]},

    so TMLE is applied twice -- once with Y as outcome and once with D
    -- both treating Z as the "treatment". The delta-method standard
    error combines the two efficient influence functions,

    .. math:: \mathrm{IF} = \frac{\mathrm{IF}_Y
              - \mathrm{LATE}\cdot \mathrm{IF}_D}{\hat\Delta_D},

    which is why running the two TMLEs and dividing the *point*
    estimates is not enough: the denominator's uncertainty has to
    enter the influence function.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment received.
    Z : array-like of {0, 1}, shape (n,)
        Instrument assigned.
    covariates : array-like, optional
        Baseline covariates.
    trunc : float, default 0.01
        Instrument-propensity truncation.

    Returns
    -------
    RichResult
        keys: ``late``, ``se``, ``ci``, ``itt``, ``compliance``,
        ``n``, ``method``.

    References
    ----------
    Tochterman, K. & van der Laan, M. J. (2011). Targeted maximum
    likelihood estimation for the LATE. UC Berkeley Division of
    Biostatistics Working Paper Series, Paper 284.

    Imbens, G. W. & Angrist, J. D. (1994). Identification and
    estimation of local average treatment effects. *Econometrica*,
    62(2), 467-475.
    """
    y = np.asarray(y, dtype=float).ravel()
    D = np.asarray(D, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    n = y.size
    if not (D.size == n and Z.size == n):
        raise ValueError("y, D, Z must have equal length.")
    for v, name in ((D, "D"), (Z, "Z")):
        if not np.all(np.isin(v, (0.0, 1.0))):
            raise ValueError(f"{name} must be binary 0/1.")
    W = np.zeros((n, 1)) if covariates is None else np.asarray(covariates, dtype=float).reshape(n, -1)

    num = tmle_ate(y, Z, W, trunc=trunc)
    den = tmle_ate(D, Z, W, trunc=trunc, scale_outcome=False)
    if abs(den["ate"]) < 1e-8:
        raise ValueError("estimated compliance is zero; the LATE is not identified.")

    late = num["ate"] / den["ate"]
    infl = (num["eif"] - late * den["eif"]) / den["ate"]
    se = float(np.sqrt((infl**2).sum()) / n)

    return RichResult(
        payload={
            "late": float(late),
            "se": se,
            "ci": (late - 1.96 * se, late + 1.96 * se),
            "itt": num["ate"],
            "compliance": den["ate"],
            "n": int(n),
            "method": "TMLE LATE (targeted ITT / targeted compliance, delta-method IF)",
        }
    )


def cheatsheet():
    return "tmlivc: TMLE(Y~Z) / TMLE(D~Z); IF = (IF_Y - LATE*IF_D) / compliance"


# compact alias per ledger/NAMING.md
tmleiv = tmle_iv
