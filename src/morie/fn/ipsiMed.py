# morie.fn -- function file (rootcoder007/morie)
"""Interventional (randomised-mediator) direct and indirect effects."""

from . import _array_core as np

from ._richresult import RichResult
from .aiptdd import _logit_fit, _ols_predict

__all__ = ["interventional_psi"]


def interventional_psi(y, x, m, c=None, n_draws=2000, seed=0):
    r"""Interventional direct and indirect effects.

    When a mediator-outcome confounder is itself affected by the
    exposure, the *natural* effects are not identified. VanderWeele,
    Vansteelandt and Robins' interventional analogues replace the
    individual counterfactual mediator :math:`M_{x'}` with a random
    draw from its population distribution given (x', C):

    .. math::
        \psi^{IDE} &= E\big[Y(x_1, G_{x_0}) - Y(x_0, G_{x_0})\big], \\
        \psi^{IIE} &= E\big[Y(x_1, G_{x_1}) - Y(x_1, G_{x_0})\big],

    with :math:`G_x \sim P(M \mid x, C)`. These are identified under
    weaker conditions and still sum to the overall effect, but they
    answer a *population* question -- "what if everyone's mediator were
    drawn from the treated distribution" -- not an individual one.

    Parameters
    ----------
    y, x, m : array-like, shape (n,)
        Outcome, binary exposure, mediator.
    c : array-like, optional
        Baseline covariates.
    n_draws : int, default 2000
        Monte Carlo draws of the randomised mediator.
    seed : int, default 0

    Returns
    -------
    RichResult
        keys: ``ide``, ``iie``, ``overall``, ``n_draws``, ``n``,
        ``method``.

    References
    ----------
    VanderWeele, T. J., Vansteelandt, S. & Robins, J. M. (2014).
    Effect decomposition in the presence of an exposure-induced
    mediator-outcome confounder. *Epidemiology*, 25(2), 300-306.
    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    m = np.asarray(m, dtype=float).ravel()
    n = y.size
    if not (x.size == n and m.size == n):
        raise ValueError("y, x, m must have equal length.")
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
            raise ValueError(f"c has {C.shape[0]} rows but y has {n}.")
    B = int(n_draws)
    if B < 100:
        raise ValueError(f"n_draws must be at least 100, got {B}.")

    # outcome model with exposure-mediator interaction
    D = np.column_stack([np.ones(n), x, m, x * m, C])
    b, *_ = np.linalg.lstsq(D, y, rcond=None)

    def predict(xv, mv):
        xc = np.full(mv.size, float(xv))
        rows = np.column_stack(
            [np.ones(mv.size), xc, mv, xc * mv, np.repeat(C.mean(axis=0)[None, :], mv.size, axis=0)]
        )
        return rows @ b

    # mediator model per arm; draw G_x from its fitted residual distribution
    rng = np.random.default_rng(seed)
    Dm = np.column_stack([np.ones(n), x, C])
    bm, *_ = np.linalg.lstsq(Dm, m, rcond=None)
    resid = m - Dm @ bm
    cbar = C.mean(axis=0)

    def draw(xv):
        mu = bm[0] + bm[1] * xv + (bm[2:] @ cbar if C.shape[1] else 0.0)
        return mu + rng.choice(resid, size=B, replace=True)

    g0, g1 = draw(0.0), draw(1.0)
    ide = float(predict(1, g0).mean() - predict(0, g0).mean())
    iie = float(predict(1, g1).mean() - predict(1, g0).mean())

    return RichResult(
        payload={
            "ide": ide,
            "iie": iie,
            "overall": ide + iie,
            "n_draws": B,
            "n": int(n),
            "method": "Interventional direct/indirect effects (randomised mediator draws)",
        }
    )


def cheatsheet():
    return "ipsiMed: IDE/IIE with M drawn from P(M | x, C) rather than the unit's own M_x"
