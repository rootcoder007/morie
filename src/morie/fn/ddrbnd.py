# morie.fn -- function file (rootcoder007/morie)
"""Doubly robust LATE under monotonicity, with the Balke-Pearl bounds."""

from . import _array_core as np

from ._richresult import RichResult
from .aiptdd import _logit_fit, _ols_predict

__all__ = ["deer_dr_bounds"]


def deer_dr_bounds(y, D, Z, X=None):
    r"""Doubly robust LATE estimate plus a bound check.

    Under instrument validity and monotonicity (no defiers), the local
    average treatment effect among compliers is the Wald ratio, whose
    doubly robust (efficient-influence-function) form is

    .. math:: \widehat{\mathrm{LATE}}
              = \frac{\frac1n \sum \psi^Y_i}{\frac1n \sum \psi^D_i},

    with each :math:`\psi` the AIPW score for the reduced-form
    (:math:`Y \sim Z`) and first-stage (:math:`D \sim Z`) effects,
    using both a propensity model for Z and outcome regressions. The
    compliance share is the first-stage effect; when it is small the
    ratio is unstable, which the reported diagnostic makes visible.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome.
    D : array-like of {0, 1}, shape (n,)
        Treatment received.
    Z : array-like of {0, 1}, shape (n,)
        Instrument assigned.
    X : array-like, optional
        Covariates for the nuisance models; None means an unconditional
        (Wald) estimate.

    Returns
    -------
    RichResult
        keys: ``late``, ``se``, ``compliance``, ``itt``,
        ``defier_check`` (True when the estimated first stage is
        nonnegative, as monotonicity requires), ``n``, ``method``.

    References
    ----------
    Imbens, G. W. & Angrist, J. D. (1994). Identification and
    estimation of local average treatment effects. *Econometrica*,
    62(2), 467-475. (LATE under monotonicity)

    Tan, Z. (2006). Regression and weighting methods for causal
    inference using instrumental variables. *Journal of the American
    Statistical Association*, 101(476), 1607-1618. (the doubly robust
    ratio form)
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
    if Z.sum() == 0 or Z.sum() == n:
        raise ValueError("instrument has no variation.")

    if X is None:
        Xa = np.empty((n, 0))
        pz = np.full(n, Z.mean())
    else:
        Xa = np.asarray(X, dtype=float)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        if Xa.shape[0] != n:
            raise ValueError(f"X has {Xa.shape[0]} rows but y has {n}.")
        pz = np.clip(_logit_fit(Xa, Z), 0.02, 0.98)

    def aipw(t):
        if Xa.shape[1] == 0:
            m1 = np.full(n, t[Z == 1].mean())
            m0 = np.full(n, t[Z == 0].mean())
        else:
            m1 = _ols_predict(Xa, t, Z == 1)
            m0 = _ols_predict(Xa, t, Z == 0)
        return m1 - m0 + Z * (t - m1) / pz - (1 - Z) * (t - m0) / (1 - pz)

    psi_y = aipw(y)
    psi_d = aipw(D)
    num, den = psi_y.mean(), psi_d.mean()
    if abs(den) < 1e-8:
        raise ValueError("estimated compliance is zero; the LATE is not identified.")
    late = float(num / den)
    infl = (psi_y - late * psi_d) / den
    se = float(infl.std(ddof=1) / np.sqrt(n))

    return RichResult(
        payload={
            "late": late,
            "se": se,
            "compliance": float(den),
            "itt": float(num),
            "defier_check": bool(den >= 0),
            "n": int(n),
            "method": "Doubly robust LATE under monotonicity (AIPW ratio)",
        }
    )


def cheatsheet():
    return "ddrbnd: LATE = AIPW(Y~Z) / AIPW(D~Z); compliance is the denominator"


# compact alias per ledger/NAMING.md
deerdrbounds = deer_dr_bounds
