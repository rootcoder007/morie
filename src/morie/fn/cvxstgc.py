# morie.fn -- function file (rootcoder007/morie)
"""Strong convexity -- Boyd & Vandenberghe Sec. 9.1.2."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["boyd_strong_convex"]


def boyd_strong_convex(f, grad_f, x, m, y_samples=None, n_probe=64,
                       radius=1.0, seed=0):
    r"""Check the strong-convexity inequality at x:

    .. math::
        f(y) \ge f(x) + \nabla f(x)^\top (y - x)
                + \frac{m}{2}\lVert y - x\rVert_2^2, \quad m > 0.

    Strong convexity is what turns "the gradient is small" into "the point
    is nearly optimal": it gives the bound
    :math:`f(x) - p^\star \le \lVert\nabla f(x)\rVert^2 / (2m)`. Without
    it a tiny gradient guarantees nothing at all -- a long flat valley has
    small gradients arbitrarily far from the minimum.

    The same m sets the convergence rate through the condition number
    :math:`M/m`, so it is not merely a certificate but the quantity that
    decides how hard the problem is.

    Parameters
    ----------
    f, grad_f : callable
        Function and gradient.
    x : array-like
        Point.
    m : float
        Strong-convexity modulus, positive.
    y_samples, n_probe, radius, seed
        Probe controls, as for :func:`~morie.fn.cvxsbp.boyd_subgradient`.

    Returns
    -------
    RichResult
        ``holds``, ``violations``, ``worst_gap``,
        ``suboptimality_bound``, ``m``.

    References
    ----------
    Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*.
        Cambridge University Press.

    Examples
    --------
    A quadratic with Hessian diag(2, 5) is 2-strongly convex, and not
    5-strongly convex.

    >>> import numpy as np
    >>> Q = np.diag([2.0, 5.0])
    >>> f = lambda z: 0.5 * z @ Q @ z
    >>> gf = lambda z: Q @ z
    >>> boyd_strong_convex(f, gf, [1.0, 1.0], m=2.0)["holds"]
    True
    >>> boyd_strong_convex(f, gf, [1.0, 1.0], m=5.5)["holds"]
    False

    The certificate: a small gradient plus strong convexity bounds the
    suboptimality, which is the whole reason to care about m.

    >>> r = boyd_strong_convex(f, gf, [0.01, 0.01], m=2.0)
    >>> bool(r["suboptimality_bound"] < 1e-3)
    True

    >>> boyd_strong_convex(f, gf, [1.0, 1.0], m=0.0)
    Traceback (most recent call last):
        ...
    ValueError: m must be positive; m = 0 is ordinary convexity, which gives no suboptimality bound
    """
    xv = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    m = float(m)
    if m <= 0:
        raise ValueError(
            "m must be positive; m = 0 is ordinary convexity, which gives "
            "no suboptimality bound")
    if y_samples is None:
        rng = np.random.default_rng(seed)
        ys = xv + rng.uniform(-radius, radius, (int(n_probe), xv.size))
        eye = np.eye(xv.size)
        ys = np.vstack([ys, xv + radius * eye, xv - radius * eye])
    else:
        ys = np.atleast_2d(np.asarray(y_samples, dtype=float))
    fx = float(f(xv))
    gx = np.atleast_1d(np.asarray(grad_f(xv), dtype=float)).ravel()
    gaps = np.empty(ys.shape[0])
    for i, y in enumerate(ys):
        d = y - xv
        gaps[i] = float(f(y)) - (fx + float(gx @ d) + 0.5 * m * float(d @ d))
    viol = int(np.sum(gaps < -1e-09))
    bound = float(gx @ gx) / (2.0 * m)
    return RichResult(
        title="Strong convexity check",
        summary_lines=[("m", m), ("probes", int(ys.shape[0])),
                       ("violations", viol),
                       ("suboptimality bound", bound)],
        warnings=["this samples the inequality locally; it can refute the "
                  "modulus but not certify it globally"],
        payload={
            "holds": bool(viol == 0), "violations": viol,
            "worst_gap": float(gaps.min()),
            "suboptimality_bound": bound, "m": m,
            "grad_norm_sq": float(gx @ gx),
            "method": "boyd_strong_convex",
        },
    )


def cheatsheet():
    return "cvxstgc: turns a small gradient into a suboptimality BOUND, ||g||^2/(2m); m also sets the rate"
