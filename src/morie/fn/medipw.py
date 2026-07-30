# morie.fn -- function file (rootcoder007/morie)
"""IPW causal mediation -- Huber (2014), JAE 29(6), 920-943."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["huber_ipw_mediation"]


def _binchoice_fit(X, y, link="probit", max_iter=100, tol=1e-09):
    """Newton-Raphson binary choice; returns fitted probabilities.

    Own solver, not a delegate: morie fits its own propensity models.
    """
    D = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.zeros(D.shape[1])
    for _ in range(int(max_iter)):
        eta = np.clip(D @ beta, -35.0, 35.0)
        if link == "probit":
            from math import erf, sqrt

            p = np.clip(0.5 * (1.0 + np.vectorize(erf)(eta / sqrt(2.0))),
                        1e-12, 1 - 1e-12)
            dens = np.exp(-0.5 * eta**2) / np.sqrt(2.0 * np.pi)
            w = dens**2 / (p * (1.0 - p))
            z = dens * (y - p) / (p * (1.0 - p))
        else:
            p = 1.0 / (1.0 + np.exp(-eta))
            w = np.maximum(p * (1.0 - p), 1e-10)
            z = y - p
        grad = D.T @ z
        H = (D * w[:, None]).T @ D
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = np.clip(D @ beta, -35.0, 35.0)
    if link == "probit":
        from math import erf, sqrt

        return 0.5 * (1.0 + np.vectorize(erf)(eta / sqrt(2.0)))
    return 1.0 / (1.0 + np.exp(-eta))


def _wmean(y, w):
    return float(np.sum(y * w) / np.sum(w))


def _medweight_point(y, d, pm, px):
    # Huber (2014) Section 3, normalised sample analogs. The weights
    # within each treatment state sum to one (Imbens 2004; Busso,
    # DiNardo & McCrary 2009), which is what the denominators do.
    y11 = _wmean(y, d / px)
    y01 = _wmean(y, (1 - d) * pm / ((1 - pm) * px))
    y10 = _wmean(y, d * (1 - pm) / (pm * (1 - px)))
    y00 = _wmean(y, (1 - d) / (1 - px))
    theta1 = y11 - y01
    theta0 = y10 - y00
    total = y11 - y00
    return {
        "total_effect": total, "direct_treated": theta1,
        "direct_control": theta0,
        "indirect_treated": total - theta0,
        "indirect_control": total - theta1,
        "y11": y11, "y01": y01, "y10": y10, "y00": y00,
    }


def huber_ipw_mediation(y, d, m, x, link="probit", trim=0.0, boot=0,
                        seed=None):
    r"""Split an average treatment effect into direct and indirect parts.

    Huber's identification needs two propensity scores,
    :math:`\Pr(D=1|X)` and :math:`\Pr(D=1|M,X)`, and imposes NO model on
    the outcome or the mediator -- arbitrary nonlinearity in either is
    allowed, which is what separates this from the
    regress-and-multiply-coefficients tradition.

    Both treatment states are reported because they genuinely differ
    whenever treatment interacts with the mediator: :math:`\theta(1)`
    holds the mediator at its treated value, :math:`\theta(0)` at its
    control value. The decomposition

    .. math::

        \Delta = \theta(1) + \delta(0) = \theta(0) + \delta(1)

    holds by construction, so it is a check on the arithmetic rather
    than a finding.

    Parameters
    ----------
    y : array-like
        Outcome.
    d : array-like
        Binary treatment (0/1).
    m : array-like
        Mediator, vector or matrix.
    x : array-like
        Covariates, ``(n, k)``.
    link : {"probit", "logit"}
        Propensity specification. Huber uses probit.
    trim : float
        Drop observations whose scores fall outside
        ``[trim, 1 - trim]``. Default 0 -- the paper's footnote 10 says
        no trimming is applied.
    boot : int
        Bootstrap replications for standard errors; 0 skips them.
    seed : int, optional
        RNG seed for the bootstrap.

    Returns
    -------
    RichResult
        ``total_effect``, ``direct_treated``, ``direct_control``,
        ``indirect_treated``, ``indirect_control``, the four weighted
        means, ``n_trimmed``, ``decomposition_holds``, ``se``.

    References
    ----------
    Huber, M. (2014). Identifying causal mechanisms (primarily) based on
        inverse probability weighting. *Journal of Applied
        Econometrics*, 29(6), 920-943.

    Examples
    --------
    Plant a direct effect of 1.0 and an indirect path of 0.8 * 1.5 = 1.2,
    so the total is 2.2.

    >>> import numpy as np
    >>> rng = np.random.default_rng(11)
    >>> n = 4000
    >>> x = rng.normal(size=(n, 2))
    >>> pr = 0.5 * (1 + np.vectorize(__import__("math").erf)(
    ...     (0.6 * x[:, 0] - 0.3 * x[:, 1]) / np.sqrt(2)))
    >>> d = (pr > rng.random(n)).astype(float)
    >>> m = 0.8 * d + 0.5 * x[:, 0] + rng.normal(size=n)
    >>> y = 1.0 * d + 1.5 * m + 0.4 * x[:, 1] + rng.normal(size=n)
    >>> r = huber_ipw_mediation(y, d, m, x)
    >>> bool(1.9 < r["total_effect"] < 2.5)
    True
    >>> bool(0.7 < r["direct_treated"] < 1.4)
    True

    The decomposition is exact, not approximate.

    >>> bool(r["decomposition_holds"])
    True

    The assertion that actually distinguishes a working estimator from
    noise: cut the treatment-to-mediator path and the indirect effect
    must collapse while the direct effect survives.

    >>> m0 = 0.5 * x[:, 0] + rng.normal(size=n)
    >>> y0 = 1.0 * d + 1.5 * m0 + rng.normal(size=n)
    >>> r0 = huber_ipw_mediation(y0, d, m0, x)
    >>> bool(abs(r0["indirect_control"]) < 0.25)
    True
    >>> bool(0.7 < r0["direct_treated"] < 1.3)
    True

    A non-binary treatment is refused: the identification is stated for
    a binary D and the weights have no meaning otherwise.

    >>> huber_ipw_mediation(y, m, m, x)
    Traceback (most recent call last):
        ...
    ValueError: d must be binary 0/1
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    d = np.atleast_1d(np.asarray(d, dtype=float)).ravel()
    m = np.atleast_2d(np.asarray(m, dtype=float))
    if m.shape[0] == 1 and m.shape[1] == y.size:
        m = m.T
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[0] == 1 and x.shape[1] == y.size:
        x = x.T
    n = y.size
    if d.size != n or m.shape[0] != n or x.shape[0] != n:
        raise ValueError(
            f"y, d, m and x must describe the same {n} observations")
    if not np.all(np.isin(d, (0.0, 1.0))):
        raise ValueError("d must be binary 0/1")
    if not (0.0 <= trim < 0.5):
        raise ValueError(f"trim must lie in [0, 0.5), got {trim}")
    if link not in ("probit", "logit"):
        raise ValueError('link must be "probit" or "logit"')
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(m))
            and np.all(np.isfinite(x))):
        raise ValueError("y, m and x must be finite")

    mx = np.column_stack([m, x])

    def fit(idx):
        px = _binchoice_fit(x[idx], d[idx], link)
        pm = _binchoice_fit(mx[idx], d[idx], link)
        keep = ((px > trim) & (px < 1 - trim)
                & (pm > trim) & (pm < 1 - trim))
        if keep.sum() < 4 or np.unique(d[idx][keep]).size < 2:
            return None
        return (_medweight_point(y[idx][keep], d[idx][keep],
                                 pm[keep], px[keep]),
                int((~keep).sum()))

    base = fit(np.arange(n))
    if base is None:
        raise ValueError(
            "no usable observations survive the common-support "
            "restriction; lower trim or check overlap")
    est, n_trimmed = base

    se = None
    if boot > 0:
        rng = np.random.default_rng(seed)
        keys = ["total_effect", "direct_treated", "direct_control",
                "indirect_treated", "indirect_control"]
        reps = []
        for _ in range(int(boot)):
            r = fit(rng.integers(0, n, n))
            if r is not None:
                reps.append([r[0][k] for k in keys])
        se = (dict(zip(keys, np.std(np.array(reps), axis=0, ddof=1)))
              if len(reps) > 1 else None)

    scale = max(1.0, abs(est["total_effect"]))
    holds = (abs(est["direct_treated"] + est["indirect_control"]
                 - est["total_effect"]) < 1e-08 * scale
             and abs(est["direct_control"] + est["indirect_treated"]
                     - est["total_effect"]) < 1e-08 * scale)
    return RichResult(
        title="IPW causal mediation",
        summary_lines=[("n", int(n)), ("link", link),
                       ("total", est["total_effect"]),
                       ("direct(1)", est["direct_treated"]),
                       ("indirect(0)", est["indirect_control"]),
                       ("trimmed", n_trimmed)],
        payload={**est, "n_trimmed": n_trimmed,
                 "decomposition_holds": bool(holds),
                 "link": link, "se": se,
                 "method": "huber_ipw_mediation"},
    )


def cheatsheet():
    return "medipw: Huber IPW mediation -- no model on Y or M; Delta = theta(1)+delta(0) = theta(0)+delta(1) exactly"
