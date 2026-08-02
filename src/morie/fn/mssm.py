# morie.fn -- function file (rootcoder007/morie)
"""Marginal structural mediation model."""

from . import _array_core as np

from ._richresult import RichResult
from .aiptdd import _logit_fit

__all__ = ["marginal_structural_med"]


def marginal_structural_med(x, m, y, c=None):
    r"""Mediation with IPW for exposure and mediator assignment.

    When the mediator-outcome relationship is confounded by the same
    baseline covariates that confound the exposure, weighting each
    unit by

    .. math:: w_i = \frac{1}{P(X_i \mid C_i)}
              \cdot \frac{1}{f(M_i \mid X_i, C_i)}

    creates a pseudo-population in which X and M are both
    unconfounded, and the *weighted* regression
    :math:`E[Y] = \theta_0 + \theta_1 X + \theta_2 M + \theta_3 XM`
    is a marginal structural model for the joint intervention.
    NDE and NIE are then read off the MSM coefficients (VanderWeele's
    weighting approach to natural effects).

    The mediator density is modelled as Gaussian given (X, C); the
    exposure must be binary.

    Parameters
    ----------
    x : array-like of {0, 1}, shape (n,)
        Binary exposure.
    m : array-like, shape (n,)
        Continuous mediator.
    y : array-like, shape (n,)
        Outcome.
    c : array-like, optional
        Baseline confounders of both X and M.

    Returns
    -------
    RichResult
        keys: ``nde``, ``nie``, ``te``, ``theta`` (the MSM
        coefficients), ``weights``, ``ess``, ``n``, ``method``.

    References
    ----------
    VanderWeele, T. J. (2009). Marginal structural models for the
    estimation of direct and indirect effects. *Epidemiology*, 20(1),
    18-26.
    """
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = x.size
    if not (m.size == n and y.size == n):
        raise ValueError("x, m, y must have equal length.")
    if not np.all(np.isin(x, (0.0, 1.0))):
        raise ValueError("x must be binary 0/1.")
    if x.sum() == 0 or x.sum() == n:
        raise ValueError("need both exposure arms.")
    if c is None:
        C = np.empty((n, 0))
    else:
        C = np.asarray(c, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"c has {C.shape[0]} rows but x has {n}.")
    if n < C.shape[1] + 8:
        raise ValueError("too few observations for the weight and outcome models.")

    # exposure weight
    e = np.clip(_logit_fit(C, x) if C.shape[1] else np.full(n, x.mean()), 0.01, 0.99)
    w_x = np.where(x == 1, 1 / e, 1 / (1 - e))

    # mediator density weight: M | X, C ~ N(mu, s2)
    D = np.column_stack([np.ones(n), x, C])
    bm, *_ = np.linalg.lstsq(D, m, rcond=None)
    res = m - D @ bm
    s2 = float((res**2).mean())
    if s2 <= 0:
        raise ValueError("mediator is perfectly predicted; density weight undefined.")
    dens = np.exp(-(res**2) / (2 * s2)) / np.sqrt(2 * np.pi * s2)
    w = w_x / np.maximum(dens, 1e-12)
    w = w / w.mean()  # ponytail: normalise; only relative weights matter for WLS

    Dy = np.column_stack([np.ones(n), x, m, x * m])
    sw = np.sqrt(w)
    theta, *_ = np.linalg.lstsq(Dy * sw[:, None], y * sw, rcond=None)
    t1, t2, t3 = float(theta[1]), float(theta[2]), float(theta[3])

    # natural effects from the MSM, mediator distribution taken at x = 0
    m0 = float(bm[0])
    b1 = float(bm[1])
    nde = t1 + t3 * m0
    nie = (t2 + t3) * b1

    return RichResult(
        payload={
            "nde": float(nde),
            "nie": float(nie),
            "te": float(nde + nie),
            "theta": theta.astype(float),
            "weights": w,
            "ess": float(w.sum() ** 2 / (w**2).sum()),
            "n": int(n),
            "method": "Marginal structural mediation model (exposure x mediator IP weights)",
        }
    )


def cheatsheet():
    return "mssm: IPW for X and M density, then MSM Y ~ X + M + XM; NDE/NIE from theta"
