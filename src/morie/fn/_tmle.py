# morie.fn -- function file (rootcoder007/morie)
"""Shared TMLE core: initial fit, clever covariate, fluctuation, EIF."""

from . import _array_core as np

from .aiptdd import _logit_fit, _ols_predict

__all__ = ["tmle_ate"]


def _logit(p):
    return np.log(p / (1 - p))


def _expit(x):
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def tmle_ate(y, A, W, trunc=0.01, max_iter=50, tol=1e-10, g=None, scale_outcome=True):
    r"""Targeted maximum likelihood estimate of the ATE.

    Three steps:

    1. **Initial fit** -- outcome regressions
       :math:`\bar Q^0(a, W) = \hat E[Y \mid A=a, W]` and the
       propensity :math:`g(W) = P(A=1 \mid W)`.
    2. **Targeting** -- fluctuate the outcome fit along the clever
       covariate

       .. math:: H(A, W) = \frac{A}{g(W)} - \frac{1-A}{1-g(W)}

       by fitting :math:`\epsilon` in
       :math:`\mathrm{logit}\,\bar Q^1 = \mathrm{logit}\,\bar Q^0
       + \epsilon H`, on the outcome rescaled to [0, 1] so the
       logistic fluctuation is valid for bounded continuous outcomes
       as well as binary ones (Gruber & van der Laan 2010).
    3. **Substitution** -- the estimate is the plug-in
       :math:`\frac1n \sum [\bar Q^1(1, W) - \bar Q^1(0, W)]`, which
       solves the efficient influence-function equation, so the EIF
       supplies the standard error directly.

    TMLE is doubly robust *and* a substitution estimator: unlike AIPW
    it can never leave the parameter space.

    Parameters
    ----------
    y : array-like, shape (n,)
        Outcome (binary or bounded continuous).
    A : array-like of {0, 1}, shape (n,)
        Treatment.
    W : array-like, shape (n, p) or (n,)
        Covariates.
    trunc : float, default 0.01
        Propensity truncation bound.
    max_iter, tol :
        Fluctuation Newton controls.
    g : array-like, optional
        Pre-computed propensity scores; skips the internal fit.
    scale_outcome : bool, default True
        Rescale y to [0, 1] for the logistic fluctuation and map back.

    Returns
    -------
    dict
        ``ate``, ``se``, ``ci`` (95%), ``eif``, ``epsilon``,
        ``q1``, ``q0`` (targeted, on the original scale), ``g``,
        ``ey1``, ``ey0``, ``n``.

    References
    ----------
    van der Laan, M. J. & Rubin, D. (2006). Targeted maximum
    likelihood learning. *The International Journal of Biostatistics*,
    2(1), Article 11.

    Gruber, S. & van der Laan, M. J. (2010). A targeted maximum
    likelihood estimator of a causal effect on a bounded continuous
    outcome. *The International Journal of Biostatistics*, 6(1),
    Article 26.
    """
    y = np.asarray(y, dtype=float).ravel()
    A = np.asarray(A, dtype=float).ravel()
    W = np.asarray(W, dtype=float)
    if W.ndim == 1:
        W = W[:, None]
    n = y.size
    if A.size != n or W.shape[0] != n:
        raise ValueError("y, A, W must share their first dimension.")
    if not np.all(np.isin(A, (0.0, 1.0))):
        raise ValueError("A must be binary 0/1.")
    if A.sum() == 0 or A.sum() == n:
        raise ValueError("need both treatment arms.")
    trunc = float(trunc)
    if not 0 <= trunc < 0.5:
        raise ValueError(f"trunc must lie in [0, 0.5), got {trunc}.")

    lo, hi = (float(y.min()), float(y.max())) if scale_outcome else (0.0, 1.0)
    rng_y = hi - lo
    if rng_y <= 0:
        raise ValueError("outcome has zero range.")
    ys = (y - lo) / rng_y if scale_outcome else y
    ys = np.clip(ys, 1e-6, 1 - 1e-6)

    if g is None:
        gw = _logit_fit(W, A)
    else:
        gw = np.asarray(g, dtype=float).ravel()
        if gw.size != n:
            raise ValueError("g must have one entry per observation.")
    gw = np.clip(gw, max(trunc, 1e-6), 1 - max(trunc, 1e-6))

    # initial outcome fit on the [0, 1] scale, clipped into the interior
    q1 = np.clip(_ols_predict(W, ys, A == 1), 1e-6, 1 - 1e-6)
    q0 = np.clip(_ols_predict(W, ys, A == 0), 1e-6, 1 - 1e-6)
    qa = np.where(A == 1, q1, q0)

    H = A / gw - (1 - A) / (1 - gw)
    H1, H0 = 1 / gw, -1 / (1 - gw)

    eps = 0.0
    off = _logit(qa)
    for _ in range(int(max_iter)):
        p = _expit(off + eps * H)
        grad = float(np.sum(H * (ys - p)))
        hess = float(np.sum(H**2 * p * (1 - p)))
        if hess <= 1e-14:
            break
        step = grad / hess
        eps += step
        if abs(step) < tol:
            break

    q1s = _expit(_logit(q1) + eps * H1)
    q0s = _expit(_logit(q0) + eps * H0)
    qas = _expit(off + eps * H)

    if scale_outcome:
        q1o, q0o, qao = q1s * rng_y + lo, q0s * rng_y + lo, qas * rng_y + lo
        yo = y
    else:
        q1o, q0o, qao, yo = q1s, q0s, qas, ys

    psi = float(np.mean(q1o - q0o))
    eif = H * (yo - qao) + (q1o - q0o) - psi
    se = float(np.sqrt((eif**2).sum()) / n)

    return {
        "ate": psi,
        "se": se,
        "ci": (psi - 1.96 * se, psi + 1.96 * se),
        "eif": eif,
        "epsilon": float(eps),
        "q1": q1o,
        "q0": q0o,
        "g": gw,
        "ey1": float(q1o.mean()),
        "ey0": float(q0o.mean()),
        "n": int(n),
    }


def cheatsheet():
    return "_tmle: initial Q and g, fluctuate along H = A/g - (1-A)/(1-g), substitute; EIF gives the SE"
