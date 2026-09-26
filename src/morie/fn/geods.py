# morie.fn -- function file (rootcoder007/morie)
"""Geodesic equation solver (Runge-Kutta on curved spacetime)."""

__all__ = ["geods"]

from . import _array_core as np
from ._sci_core import solve_ivp


def geods(
    metric_func,
    x0: np.ndarray,
    u0: np.ndarray,
    tau_span: tuple = (0.0, 10.0),
    n_points: int = 500,
    h: float = 1e-5,
) -> dict:
    r"""
    Solve the geodesic equation via numerical integration.

    .. math::

        \\frac{d^2 x^\\mu}{d\\tau^2}
        + \\Gamma^\\mu_{\\alpha\\beta}
          \\frac{dx^\\alpha}{d\\tau}\\frac{dx^\\beta}{d\\tau} = 0

    Christoffel symbols are computed numerically from the metric.

    Parameters
    ----------
    metric_func : callable
        metric_func(x) -> (4,4) ndarray, the metric at position x.
    x0 : np.ndarray
        Initial 4-position [t, r, theta, phi] (or similar coords).
    u0 : np.ndarray
        Initial 4-velocity.
    tau_span : tuple
        (tau_start, tau_end) proper time range.
    n_points : int
        Number of output points.
    h : float
        Step size for numerical differentiation of metric.

    Returns
    -------
    dict
        Keys: tau (1-d), position (n,4), velocity (n,4).
    """
    x0 = np.asarray(x0, dtype=float)
    u0 = np.asarray(u0, dtype=float)
    if x0.shape != (4,) or u0.shape != (4,):
        raise ValueError("x0 and u0 must be length-4.")

    def _mat(x):
        return [[float(v) for v in row] for row in np.asarray(metric_func(x), dtype=float).tolist()]

    def christoffel(x):
        # Gamma^lam_{mu nu} = 1/2 g^{lam sig} (d_mu g_{sig nu} + d_nu g_{sig mu}
        # - d_sig g_{mu nu}), metric derivatives by central differences;
        # plain lists keep the RK45 inner loop free of array indexing
        xl = [float(v) for v in x]
        ginv = [[float(v) for v in row] for row in np.linalg.inv(np.asarray(_mat(xl))).tolist()]
        dg = []
        for mu in range(4):
            xp = list(xl)
            xm = list(xl)
            xp[mu] += h
            xm[mu] -= h
            gp, gm = _mat(xp), _mat(xm)
            dg.append([[(gp[i][j] - gm[i][j]) / (2.0 * h) for j in range(4)] for i in range(4)])
        return [[[0.5 * sum(ginv[lam][sig] * (dg[mu][sig][nu] + dg[nu][sig][mu] - dg[sig][mu][nu])
                            for sig in range(4))
                  for nu in range(4)] for mu in range(4)] for lam in range(4)]

    def rhs(tau, y):
        yl = [float(v) for v in y]
        x, u = yl[:4], yl[4:]
        G = christoffel(x)
        accel = [-sum(G[mu][a][b] * u[a] * u[b] for a in range(4) for b in range(4))
                 for mu in range(4)]
        return np.asarray(u + accel, dtype=float)

    y0 = np.concatenate([x0, u0])
    tau_eval = np.linspace(tau_span[0], tau_span[1], n_points)
    sol = solve_ivp(rhs, tau_span, y0, t_eval=tau_eval, method="RK45", rtol=1e-10, atol=1e-12)

    return {
        "tau": sol.t,
        "position": sol.y[:4].T,
        "velocity": sol.y[4:].T,
    }


def cheatsheet() -> str:
    return "geods: geods(metric_func, x0, u0, tau_span, n_points, h) -> Solve the geodesic equation via numerical integration."
