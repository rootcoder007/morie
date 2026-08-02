# morie.fn -- function file (rootcoder007/morie)
"""Shared bivariate-copula core: CDFs, Kendall's tau, dependence measures.

Parameter/tau relations follow Czado (2019), *Analyzing Dependent Data
with Vine Copulas*, Table 3.2 p. 54 -- read in the library PDF, not
recalled. Theorem 3.9 eq. (3.17) gives the general Archimedean
integral; the closed forms below are that table's rows.
"""

from . import _array_core as np
from ._sci_core import integrate
from ._sci_core import special
from . import _stats_core as stats

__all__ = ["copula_cdf", "copula_tau", "tau_to_theta", "FAMILIES"]

FAMILIES = ("independence", "gaussian", "t", "clayton", "gumbel", "frank", "joe", "plackett")


def _uv(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        u, v = np.broadcast_arrays(u, v)
    if np.any((u < 0) | (u > 1)) or np.any((v < 0) | (v > 1)):
        raise ValueError("u and v must lie in [0, 1].")
    return np.clip(u, 1e-12, 1 - 1e-12), np.clip(v, 1e-12, 1 - 1e-12)


def copula_cdf(family, u, v, theta=None, nu=None):
    r"""Bivariate copula CDF :math:`C(u, v)`.

    Families and parameter ranges (Czado 2019 Ch. 3):

    - ``independence``: :math:`uv`
    - ``gaussian``: :math:`\Phi_\rho(\Phi^{-1}u, \Phi^{-1}v)`,
      :math:`\rho \in (-1, 1)`
    - ``t``: :math:`T_{\nu,\rho}(t_\nu^{-1}u, t_\nu^{-1}v)`
    - ``clayton``: :math:`(u^{-\theta} + v^{-\theta} - 1)^{-1/\theta}`,
      :math:`\theta > 0`
    - ``gumbel``: :math:`\exp\{-[(-\ln u)^\theta
      + (-\ln v)^\theta]^{1/\theta}\}`, :math:`\theta \ge 1`
    - ``frank``: :math:`-\tfrac1\theta \ln\!\big(1 +
      \tfrac{(e^{-\theta u}-1)(e^{-\theta v}-1)}{e^{-\theta}-1}\big)`,
      :math:`\theta \ne 0`
    - ``joe``: :math:`1 - [\bar u^\theta + \bar v^\theta
      - \bar u^\theta \bar v^\theta]^{1/\theta}`, :math:`\theta \ge 1`
    - ``plackett``: the cross-product-ratio family, :math:`\theta > 0`

    Every family satisfies the copula boundary conditions
    :math:`C(u, 1) = u`, :math:`C(1, v) = v`, :math:`C(u, 0) = 0`,
    which the tests check family by family.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}.")
    u, v = _uv(u, v)

    if family == "independence":
        return u * v
    if family == "gaussian":
        rho = float(theta)
        if not -1 < rho < 1:
            raise ValueError(f"gaussian rho must lie in (-1, 1), got {rho}.")
        x, y = stats.norm.ppf(u), stats.norm.ppf(v)
        out = np.empty(np.shape(x))
        flat_x, flat_y = np.atleast_1d(x).ravel(), np.atleast_1d(y).ravel()
        vals = np.array(
            [
                stats.multivariate_normal.cdf([xi, yi], mean=[0, 0], cov=[[1, rho], [rho, 1]])
                for xi, yi in zip(flat_x, flat_y)
            ]
        )
        return vals.reshape(np.shape(x)) if np.ndim(x) else float(vals[0])
    if family == "t":
        rho = float(theta)
        if not -1 < rho < 1:
            raise ValueError(f"t copula rho must lie in (-1, 1), got {rho}.")
        if nu is None or float(nu) <= 0:
            raise ValueError("t copula needs nu > 0.")
        nu = float(nu)
        x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
        flat_x, flat_y = np.atleast_1d(x).ravel(), np.atleast_1d(y).ravel()
        vals = np.array([_bt_cdf(xi, yi, rho, nu) for xi, yi in zip(flat_x, flat_y)])
        return vals.reshape(np.shape(x)) if np.ndim(x) else float(vals[0])
    if family == "clayton":
        th = float(theta)
        if th <= 0:
            raise ValueError(f"clayton theta must be positive, got {th}.")
        return np.maximum(u ** (-th) + v ** (-th) - 1.0, 1e-300) ** (-1.0 / th)
    if family == "gumbel":
        th = float(theta)
        if th < 1:
            raise ValueError(f"gumbel theta must be >= 1, got {th}.")
        a = (-np.log(u)) ** th + (-np.log(v)) ** th
        return np.exp(-(a ** (1.0 / th)))
    if family == "frank":
        th = float(theta)
        if th == 0:
            raise ValueError("frank theta must be non-zero (theta -> 0 is independence).")
        num = np.expm1(-th * u) * np.expm1(-th * v)
        return -np.log1p(num / np.expm1(-th)) / th
    if family == "joe":
        th = float(theta)
        if th < 1:
            raise ValueError(f"joe theta must be >= 1, got {th}.")
        ub, vb = (1 - u) ** th, (1 - v) ** th
        return 1.0 - (ub + vb - ub * vb) ** (1.0 / th)
    # plackett
    th = float(theta)
    if th <= 0:
        raise ValueError(f"plackett theta must be positive, got {th}.")
    if np.isclose(th, 1.0):
        return u * v
    eta = th - 1.0
    s = 1.0 + eta * (u + v)
    return (s - np.sqrt(s**2 - 4.0 * th * eta * u * v)) / (2.0 * eta)


def _bt_cdf(x, y, rho, nu):
    """Bivariate t CDF by one-dimensional quadrature over the chi mixing variable."""
    if not np.isfinite(x) or not np.isfinite(y):
        return float(np.isfinite(x) and x > 0) * float(np.isfinite(y) and y > 0)

    def inner(w):
        # given the scale mixture, (x, y) sqrt(nu/w) is bivariate normal
        s = np.sqrt(w / nu)
        return stats.multivariate_normal.cdf(
            [x * s, y * s], mean=[0, 0], cov=[[1, rho], [rho, 1]]
        ) * stats.chi2.pdf(w, nu)

    val, _ = integrate.quad(inner, 1e-8, nu + 12 * np.sqrt(2 * nu), limit=120)
    return float(val)


def copula_tau(family, theta=None, nu=None):
    r"""Kendall's tau from the copula parameter (Czado 2019 Table 3.2, p. 54).

    - gaussian, t: :math:`\tau = \tfrac2\pi \arcsin\rho`
    - gumbel: :math:`\tau = 1 - 1/\delta`
    - clayton: :math:`\tau = \delta/(\delta + 2)`
    - frank: :math:`\tau = 1 - \tfrac4\delta + \tfrac4{\delta}D_1(\delta)`
      with the Debye function
      :math:`D_1(\delta) = \tfrac1\delta\int_0^\delta \tfrac{x}{e^x-1}dx`
    - joe: the digamma expression of Table 3.2
    - independence: 0

    Plackett has no closed form in that table, so it is computed from
    the general Archimedean/copula double integral -- and the
    docstring says so rather than inventing a formula.
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}.")
    if family == "independence":
        return 0.0
    if family in ("gaussian", "t"):
        rho = float(theta)
        if not -1 < rho < 1:
            raise ValueError("rho must lie in (-1, 1).")
        return float(2.0 / np.pi * np.arcsin(rho))
    if family == "gumbel":
        d = float(theta)
        if d < 1:
            raise ValueError("gumbel theta must be >= 1.")
        return float(1.0 - 1.0 / d)
    if family == "clayton":
        d = float(theta)
        if d <= 0:
            raise ValueError("clayton theta must be positive.")
        return float(d / (d + 2.0))
    if family == "frank":
        d = float(theta)
        if d == 0:
            raise ValueError("frank theta must be non-zero.")
        # Debye function of order 1, D_1(d) = (1/d) int_0^d x/(e^x - 1) dx
        f = lambda x: x / np.expm1(x) if x != 0 else 1.0
        val, _ = integrate.quad(f, 0.0, abs(d), limit=200)
        D1 = val / abs(d)
        tau = 1.0 - 4.0 / abs(d) * (1.0 - D1)
        return float(tau if d > 0 else -tau)
    if family == "joe":
        d = float(theta)
        if d < 1:
            raise ValueError("joe theta must be >= 1.")
        if np.isclose(d, 1.0):
            return 0.0
        g = 0.5772156649015329  # Euler-Mascheroni
        return float(
            1.0
            + (-2.0 + 2.0 * g + 2.0 * np.log(2.0) + special.digamma(1.0 / d)
               + special.digamma(0.5 * (2.0 + d) / d) + d)
            / (-2.0 + d)
        )
    # plackett: numeric, no closed form in Table 3.2
    return _tau_numeric(family, theta, nu)


def _tau_numeric(family, theta=None, nu=None, n=200):
    r"""tau = 4 int int C dC - 1, evaluated on a grid via the copula density.

    Uses the identity :math:`\tau = 4\int\int C\,dC - 1` with the
    density approximated by second differences of C on an n x n grid --
    accurate to O(1/n) and used only for families whose closed form
    Czado's Table 3.2 does not give.
    """
    g = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(g, g, indexing="ij")
    h = 1.0 / (2 * n)
    C = copula_cdf(family, U, V, theta, nu)
    dens = (
        copula_cdf(family, U + h, V + h, theta, nu)
        - copula_cdf(family, U + h, np.clip(V - h, 1e-12, 1), theta, nu)
        - copula_cdf(family, np.clip(U - h, 1e-12, 1), V + h, theta, nu)
        + copula_cdf(family, np.clip(U - h, 1e-12, 1), np.clip(V - h, 1e-12, 1), theta, nu)
    ) / (4 * h * h)
    return float(4.0 * np.sum(C * dens) / n**2 - 1.0)


def tau_to_theta(family, tau):
    """Invert the tau relation for the one-parameter families."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}.")
    tau = float(tau)
    if not -1 < tau < 1:
        raise ValueError(f"tau must lie in (-1, 1), got {tau}.")
    if family == "independence":
        return None
    if family in ("gaussian", "t"):
        return float(np.sin(np.pi * tau / 2.0))
    if family == "gumbel":
        if tau <= 0:
            raise ValueError("gumbel admits only tau > 0.")
        return float(1.0 / (1.0 - tau))
    if family == "clayton":
        if tau <= 0:
            raise ValueError("clayton admits only tau > 0.")
        return float(2.0 * tau / (1.0 - tau))
    # frank, joe, plackett: monotone in theta, invert by bisection
    from ._sci_core import optimize

    lo, hi = (1e-6, 60.0) if family != "frank" else (1e-6, 60.0)
    if family == "joe":
        lo = 1.0 + 1e-9
    if family == "frank" and tau < 0:
        lo, hi = -60.0, -1e-6
    f = lambda t: copula_tau(family, t) - tau
    return float(optimize.brentq(f, lo, hi, xtol=1e-10))


def cheatsheet():
    return "_copula: CDFs + Czado Table 3.2 tau relations + tau->theta inversion"
