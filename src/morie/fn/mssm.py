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

    .. math:: w_i = \frac{P(X_i)}{P(X_i \mid C_i)}
              \cdot \frac{f(M_i \mid X_i)}{f(M_i \mid X_i, C_i)}

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

    # stabilized weights (VanderWeele 2009, eq. 3):
    #   sw = P(X) / P(X | C)  *  f(M | X) / f(M | X, C)
    # The numerators keep the pseudo-population the size of the sample
    # and leave the MSM coefficients unchanged in expectation; without
    # them 1 / f(M | X, C) explodes in the Gaussian tails and the
    # estimates swing wildly from sample to sample.
    e = np.clip(_logit_fit(C, x) if C.shape[1] else np.full(n, x.mean()), 0.01, 0.99)
    px = float(x.mean())
    w_x = np.where(x == 1, px / e, (1 - px) / (1 - e))

    def _gauss_fit(Dm):
        b, *_ = np.linalg.lstsq(Dm, m, rcond=None)
        r = m - Dm @ b
        v = float((r**2).mean())
        if v <= 0:
            raise ValueError("mediator is perfectly predicted; density weight undefined.")
        return b, np.exp(-(r**2) / (2 * v)) / np.sqrt(2 * np.pi * v)

    D = np.column_stack([np.ones(n), x, C])
    bm, dens_xc = _gauss_fit(D)
    _, dens_x = _gauss_fit(np.column_stack([np.ones(n), x]))
    w = w_x * dens_x / np.maximum(dens_xc, 1e-300)
    Dy = np.column_stack([np.ones(n), x, m, x * m])
    sw = np.sqrt(w)
    theta, *_ = np.linalg.lstsq(Dy * sw[:, None], y * sw, rcond=None)
    t1, t2, t3 = float(theta[1]), float(theta[2]), float(theta[3])

    # natural effects from the MSM: E[M_0] averages the x = 0 mediator
    # model over the covariate distribution, not at C = 0
    D0 = np.column_stack([np.ones(n), np.zeros(n), C])
    m0 = float(np.mean(D0 @ bm))
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
