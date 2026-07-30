# morie.fn -- function file (rootcoder007/morie)
"""Normalizing-flow density estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["normalizing_flow_density", "normalizing_flow"]


def normalizing_flow_density(x, at=None, n_layers=8, n_iter=400, lr=0.05,
                             seed=0):
    r"""Density by a chain of invertible maps, fitted by exact likelihood.

    A flow writes the density through the change of variables

    .. math::
       \log p(x) = \log p_Z(f(x))
                 + \log\left|\det \frac{\partial f}{\partial x}\right|,

    with :math:`f` a composition of invertible maps and :math:`p_Z`
    a standard normal. Because the map is invertible and its Jacobian
    tractable, the likelihood is EXACT -- no bound, no sampling
    estimate. That is the property that separates flows from VAEs
    (which optimise a lower bound) and GANs (which have no likelihood
    at all).

    The implementation here is a chain of elementwise affine and
    :math:`\tanh` maps, which is deliberately modest: it is enough to
    fit skew and light multimodality in one dimension while keeping
    every Jacobian diagonal, so the log-determinant is a sum rather
    than a decomposition. Richer couplings buy expressiveness at the
    cost of that transparency.

    ``log_likelihood`` is comparable across models on the same data,
    which is the flow's main practical advantage over a KDE: bandwidth
    selection has no likelihood to compare, while flows can be
    model-selected directly. ``integral`` checks numerically that the
    fitted density integrates to one -- if the Jacobian term were wrong
    it would not, and nothing else would reveal it.

    Parameters
    ----------
    x : array-like, shape (n,)
    at : array-like, optional
    n_layers, n_iter, lr : int, int, float
    seed : int

    Returns
    -------
    RichResult
        ``density``, ``at``, ``log_likelihood``, ``integral``,
        ``aic``, ``n_parameters``, ``converged``.

    References
    ----------
    Rezende and Mohamed (2015), "Variational inference with
    normalizing flows", ICML, arXiv:1505.05770.
    Papamakarios et al. (2021), *JMLR* 22:1-64, for the survey.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> out = normalizing_flow_density(rng.normal(size=300), n_iter=60)
    >>> bool(out["integral"] > 0.8)
    True
    """
    v = np.asarray(x, dtype=float).ravel()
    n = v.size
    if n < 5:
        raise ValueError("need at least 5 observations, got %d." % n)
    if np.any(~np.isfinite(v)):
        raise ValueError("x contains non-finite values.")
    L = int(n_layers)
    if L < 1:
        raise ValueError("n_layers must be at least 1.")

    mu, sd = float(v.mean()), float(v.std(ddof=1)) or 1.0
    z0 = (v - mu) / sd
    rng = np.random.default_rng(int(seed))
    a = np.zeros(L)                       # log-scale per layer
    b = np.zeros(L)                       # shift per layer
    w = rng.normal(scale=0.01, size=L)    # tanh mixing weight

    def forward(u, a, b, w):
        ld = np.zeros_like(u)
        for k in range(L):
            s = np.exp(np.clip(a[k], -5, 5))
            u = s * u + b[k]
            ld = ld + a[k]
            t = np.tanh(u)
            u = u + w[k] * t
            ld = ld + np.log(np.abs(1.0 + w[k] * (1.0 - t ** 2)) + 1e-12)
        return u, ld

    def nll(a, b, w):
        u, ld = forward(z0, a, b, w)
        lp = -0.5 * u ** 2 - 0.5 * np.log(2 * np.pi)
        return -float(np.mean(lp + ld))

    cur = nll(a, b, w)
    step = float(lr)
    hist = [cur]
    for _ in range(int(n_iter)):
        ga = np.zeros(L); gb = np.zeros(L); gw = np.zeros(L)
        h = 1e-5
        for k in range(L):
            for arr, g in ((a, ga), (b, gb), (w, gw)):
                old = arr[k]
                arr[k] = old + h
                up = nll(a, b, w)
                arr[k] = old - h
                dn = nll(a, b, w)
                arr[k] = old
                g[k] = (up - dn) / (2 * h)
        a2, b2, w2 = a - step * ga, b - step * gb, w - step * gw
        w2 = np.clip(w2, -0.95, 0.95)      # keeps the map invertible
        new = nll(a2, b2, w2)
        if new < cur:
            a, b, w, cur = a2, b2, w2, new
            step *= 1.1
        else:
            step *= 0.5
            if step < 1e-8:
                break
        hist.append(cur)

    grid = (np.linspace(v.min() - 3 * sd, v.max() + 3 * sd, 400)
            if at is None else np.asarray(at, dtype=float).ravel())
    zq = (grid - mu) / sd
    u, ld = forward(zq, a, b, w)
    logp = -0.5 * u ** 2 - 0.5 * np.log(2 * np.pi) + ld - np.log(sd)
    dens = np.exp(logp)
    integral = float(np.trapezoid(dens, grid)) if grid.size > 1 else np.nan
    npar = 3 * L
    ll = -cur * n
    return RichResult(
        payload={
            "estimate": dens,
            "density": dens,
            "at": grid,
            "log_density": logp,
            "log_likelihood": float(ll),
            "aic": float(-2 * ll + 2 * npar),
            "bic": float(-2 * ll + npar * np.log(n)),
            "n_parameters": int(npar),
            "integral": integral,
            "integral_note": (
                "a numeric check on the change-of-variables term: a wrong "
                "log-Jacobian gives a density that does not integrate to "
                "one, and nothing else in the fit would show it"
            ),
            "exact_likelihood_note": (
                "the likelihood is exact rather than a bound, which is what "
                "lets flows be model-selected directly; a KDE has no "
                "likelihood to compare bandwidths with"
            ),
            "loss_history": np.asarray(hist),
            "converged": bool(len(hist) > 2
                              and abs(hist[-1] - hist[-2]) < 1e-8),
            "n_layers": L,
            "n": int(n),
            "method": "Normalizing-flow density estimate",
        }
    )


def cheatsheet():
    return (
        "normfl: 1-D normalizing flow with exact likelihood and a "
        "unit-integral check on the Jacobian term"
    )


#: Catalogue alias for :func:`normalizing_flow_density`.
normalizing_flow = normalizing_flow_density
