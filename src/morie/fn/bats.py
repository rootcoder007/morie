# morie.fn -- function file (rootcoder007/morie)
r"""BATS and TBATS: exponential smoothing for complex seasonal patterns.

De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2010) "Forecasting
time series with complex seasonal patterns using exponential smoothing",
Monash University Department of Econometrics and Business Statistics
Working Paper 15/10, 28 October 2010 -- the working version of *Journal
of the American Statistical Association* 106(496), 1513-1527 (2011).

Classical exponential smoothing carries one seasonal index per period,
which fails on three kinds of series the paper opens with: a weekly
series whose annual period is the non-integer 365.25/7, call-centre
arrivals with nested daily and weekly periods of 169 and 845, and
Turkish electricity demand with two *non-nested* annual periods, the
Gregorian 365.25 and the Hijri 354.37.

**BATS** -- Box-Cox transform, ARMA errors, Trend, Seasonal -- is
equations 3a-3f:

.. math::

   y_t^{(\omega)} &= \begin{cases}
       (y_t^{\omega} - 1)/\omega, & \omega \neq 0 \\
       \log y_t, & \omega = 0 \end{cases} \\
   y_t^{(\omega)} &= \ell_{t-1} + \phi b_{t-1}
        + \sum_{i=1}^{T} s^{(i)}_{t-m_i} + d_t \\
   \ell_t &= \ell_{t-1} + \phi b_{t-1} + \alpha d_t \\
   b_t &= (1 - \phi) b + \phi b_{t-1} + \beta d_t \\
   s^{(i)}_t &= s^{(i)}_{t-m_i} + \gamma_i d_t \\
   d_t &= \sum_{i=1}^{p} \varphi_i d_{t-i}
        + \sum_{i=1}^{q} \theta_i \varepsilon_{t-i} + \varepsilon_t.

Two details are the authors' own and are easy to get wrong. The trend
is the Gardner & McKenzie damped trend but supplemented, following
Snyder (2006), with a **long-run** trend :math:`b`, so forecasts of
:math:`b_t` converge to :math:`b` rather than to zero. And
:math:`\phi` appears in the level and measurement equations as well as
the trend equation -- for consistency with Gardner & McKenzie; the
predictions are identical if it is left out of those two.

The model is written ``BATS(omega, phi, p, q, m1, ..., mT)``. The paper
names the special cases: ``BATS(1, 1, 0, 0, m1)`` is Holt-Winters'
additive method, ``BATS(1, 1, 0, 0, m1, m2)`` is Taylor's (2003) double
seasonal, and ``BATS(1, 1, 1, 0, m1, m2)`` is Taylor's with the AR(1)
residual adjustment.

**TBATS** replaces the seasonal index with a trigonometric one,
equations 4a-4c:

.. math::

   s^{(i)}_t &= \sum_{j=1}^{k_i} s^{(i)}_{j,t} \\
   s^{(i)}_{j,t} &= s^{(i)}_{j,t-1} \cos \lambda^{(i)}_j
       + s^{*(i)}_{j,t-1} \sin \lambda^{(i)}_j + \gamma^{(i)}_1 d_t \\
   s^{*(i)}_{j,t} &= -s^{(i)}_{j,t-1} \sin \lambda^{(i)}_j
       + s^{*(i)}_{j,t-1} \cos \lambda^{(i)}_j + \gamma^{(i)}_2 d_t,

with :math:`\lambda^{(i)}_j = 2\pi j / m_i`. This buys three things:
:math:`2\sum_i k_i` seed values instead of :math:`\sum_i m_i`, a
**non-integer** :math:`m_i` becomes expressible, and setting the
smoothing parameters to zero gives deterministic seasonality. The paper
states the equivalence that pins the parameterisation down: the
trigonometric form reproduces the index form when
:math:`k_i = m_i / 2` for even :math:`m_i` and
:math:`k_i = (m_i - 1)/2` for odd.

**Estimation.** With :math:`\varepsilon_t \sim N(0, \sigma^2)` the
density of the original series picks up the Box-Cox Jacobian
:math:`\prod_t y_t^{\omega - 1}`, so concentrating out :math:`\sigma^2`
leaves

.. math::

   L^{*} = -\frac{n}{2} \log \sum_t \varepsilon_t^2
           + (\omega - 1) \sum_t \log y_t.

The seed state :math:`x_0` is concentrated out too: the residuals are
affine in :math:`x_0`, so it is a linear least-squares solve rather
than something the optimiser has to search, which is what the paper
does and what it credits with better forecasts and shorter runs.
Only the small vector (:math:`\omega, \phi, \alpha, \beta, \gamma,
\varphi, \theta`) is searched.
"""

import math

from . import _array_core as np

from ._richresult import RichResult
from ._sci_core import minimize

__all__ = [
    "bats",
    "tbats",
    "box_cox",
    "inv_box_cox",
    "bats_filter",
    "seasonal_harmonics",
    "concentrated_loglik",
    "fit_seed_state",
    "state_matrices",
    "spectral_radius",
    "all_eigenvalues",
    "is_forecastable",
    "BatsSpec",
]


# --------------------------------------------------------------------------
# the Box-Cox transform
# --------------------------------------------------------------------------

def box_cox(y, omega):
    r"""Equation 3a. ``omega == 0`` is the log, exactly, not a limit."""
    y = [float(v) for v in y]
    if any(v <= 0.0 for v in y):
        raise ValueError("bats: the Box-Cox transform needs a strictly "
                         "positive series")
    if omega == 0.0:
        return [math.log(v) for v in y]
    return [(v ** omega - 1.0) / omega for v in y]


def inv_box_cox(z, omega):
    """The inverse of :func:`box_cox`."""
    z = [float(v) for v in z]
    if omega == 0.0:
        return [math.exp(v) for v in z]
    out = []
    for v in z:
        base = omega * v + 1.0
        if base <= 0.0:
            raise ValueError("bats: the inverse Box-Cox transform is "
                             "undefined here (omega*z + 1 <= 0)")
        out.append(base ** (1.0 / omega))
    return out


def seasonal_harmonics(m, k=None):
    r"""The :math:`\lambda_j = 2\pi j/m` of equation 4c.

    ``k`` defaults to the paper's index-equivalent count:
    :math:`m/2` for even :math:`m`, :math:`(m-1)/2` for odd. A
    non-integer ``m`` is allowed, which is the point of the
    trigonometric form.
    """
    m = float(m)
    if m <= 1.0:
        raise ValueError("bats: a seasonal period must exceed 1")
    if k is None:
        mi = int(round(m))
        if abs(m - mi) < 1e-9:
            k = mi // 2 if mi % 2 == 0 else (mi - 1) // 2
        else:
            k = int(math.floor(m / 2.0))
    k = int(k)
    if k < 1:
        raise ValueError("bats: a seasonal component needs at least one "
                         "harmonic")
    if k > m / 2.0 + 1e-9:
        raise ValueError("bats: k = %d exceeds m/2 = %g; the harmonics "
                         "above m/2 are aliases of those below" % (k, m / 2.0))
    return [2.0 * math.pi * (j + 1) / m for j in range(k)]


# --------------------------------------------------------------------------
# the model specification
# --------------------------------------------------------------------------

class BatsSpec(object):
    r"""``BATS(omega, phi, p, q, m1, ..., mT)`` or its TBATS form.

    ``harmonics`` is ``None`` for the index seasonal of equation 3e, or
    a list of :math:`k_i` for the trigonometric seasonal of 4a-4c.
    """

    def __init__(self, periods=(), harmonics=None, use_box_cox=False,
                 use_trend=True, damped=False, p=0, q=0):
        self.periods = [float(v) for v in periods]
        for m in self.periods:
            if m <= 1.0:
                raise ValueError("bats: a seasonal period must exceed 1")
        if harmonics is None:
            self.harmonics = None
            for m in self.periods:
                if abs(m - round(m)) > 1e-9:
                    raise ValueError(
                        "bats: the index seasonal of eq. 3e needs integer "
                        "periods; m = %g is not one. Use the trigonometric "
                        "seasonal (harmonics=...) for a fractional period."
                        % m)
        else:
            if len(harmonics) != len(self.periods):
                raise ValueError("bats: %d harmonic counts for %d periods"
                                 % (len(harmonics), len(self.periods)))
            self.harmonics = [len(seasonal_harmonics(m, k))
                              for m, k in zip(self.periods, harmonics)]
        self.use_box_cox = bool(use_box_cox)
        self.use_trend = bool(use_trend)
        self.damped = bool(damped) and self.use_trend
        self.p = int(p)
        self.q = int(q)
        if self.p < 0 or self.q < 0:
            raise ValueError("bats: p and q must be non-negative")

    @property
    def trigonometric(self):
        return self.harmonics is not None

    def n_states(self):
        """Level, short-run trend, seasonals, and the ARMA lags."""
        n = 1 + (1 if self.use_trend else 0)
        if self.trigonometric:
            n += 2 * sum(self.harmonics)
        else:
            n += int(sum(round(m) for m in self.periods))
        return n + self.p + self.q

    def n_free(self):
        """The parameters the optimiser searches."""
        n = 1                                   # alpha
        if self.use_trend:
            n += 1                              # beta
            if self.damped:
                n += 1                          # phi
        if self.trigonometric:
            n += 2 * len(self.periods)          # gamma_1, gamma_2 each
        else:
            n += len(self.periods)              # gamma_i
        n += self.p + self.q
        if self.use_box_cox:
            n += 1
        return n

    def label(self):
        head = "TBATS" if self.trigonometric else "BATS"
        omega = "omega" if self.use_box_cox else "1"
        phi = "phi" if self.damped else "1"
        if self.trigonometric:
            seas = ", ".join("{%g, %d}" % (m, k) for m, k
                             in zip(self.periods, self.harmonics))
        else:
            seas = ", ".join("%g" % m for m in self.periods)
        parts = [omega, phi, str(self.p), str(self.q)]
        return "%s(%s%s)" % (head, ", ".join(parts),
                             (", " + seas) if seas else "")


# --------------------------------------------------------------------------
# the recursion
# --------------------------------------------------------------------------

def _unpack(spec, theta):
    """Split the free vector into named parameters."""
    i = 0
    alpha = theta[i]
    i += 1
    beta = 0.0
    phi = 1.0
    if spec.use_trend:
        beta = theta[i]
        i += 1
        if spec.damped:
            phi = theta[i]
            i += 1
    if spec.trigonometric:
        g1 = []
        g2 = []
        for _ in spec.periods:
            g1.append(theta[i])
            g2.append(theta[i + 1])
            i += 2
        gam = (g1, g2)
    else:
        gam = [theta[i + j] for j in range(len(spec.periods))]
        i += len(spec.periods)
    ar = [theta[i + j] for j in range(spec.p)]
    i += spec.p
    ma = [theta[i + j] for j in range(spec.q)]
    i += spec.q
    omega = theta[i] if spec.use_box_cox else 1.0
    return alpha, beta, phi, gam, ar, ma, omega


def bats_filter(z, spec, theta, x0, long_run_b=0.0):
    r"""Run equations 3b-3f (or 4a-4c) and return the innovations.

    ``z`` is the **already transformed** series :math:`y^{(\omega)}`;
    ``x0`` is the seed state, laid out as level, short-run trend, the
    seasonal states, then the :math:`p` lagged :math:`d` and the
    :math:`q` lagged :math:`\varepsilon`.

    Returns ``(residuals, fitted, carry)``. ``fitted`` is
    :math:`w' x_{t-1}`, the one-step-ahead prediction on the
    transformed scale, and ``carry`` is the state after the last
    observation, which is what a forecast starts from.
    """
    alpha, beta, phi, gam, ar, ma, _ = _unpack(spec, theta)
    n = len(z)
    lev = float(x0[0])
    i = 1
    trend = 0.0
    if spec.use_trend:
        trend = float(x0[i])
        i += 1

    if spec.trigonometric:
        lam = [seasonal_harmonics(m, k)
               for m, k in zip(spec.periods, spec.harmonics)]
        s = []
        sstar = []
        for ki in spec.harmonics:
            s.append([float(x0[i + j]) for j in range(ki)])
            i += ki
            sstar.append([float(x0[i + j]) for j in range(ki)])
            i += ki
    else:
        lam = None
        buf = []
        for m in spec.periods:
            mi = int(round(m))
            buf.append([float(x0[i + j]) for j in range(mi)])
            i += mi

    dlag = [float(x0[i + j]) for j in range(spec.p)]
    i += spec.p
    elag = [float(x0[i + j]) for j in range(spec.q)]

    resid = []
    fitted = []
    for t in range(n):
        if spec.trigonometric:
            seas = sum(sum(s[a]) for a in range(len(spec.periods)))
        else:
            seas = sum(buf[a][0] for a in range(len(spec.periods)))
        # w' x_{t-1} plus the ARMA part of d_t that is already known
        base = lev + phi * trend + seas
        darma = (sum(ar[j] * dlag[j] for j in range(spec.p))
                 + sum(ma[j] * elag[j] for j in range(spec.q)))
        pred = base + darma
        eps = z[t] - pred
        d = darma + eps
        fitted.append(pred)
        resid.append(eps)

        new_lev = lev + phi * trend + alpha * d
        new_trend = ((1.0 - phi) * long_run_b + phi * trend + beta * d
                     if spec.use_trend else 0.0)
        if spec.trigonometric:
            for a in range(len(spec.periods)):
                g1 = gam[0][a]
                g2 = gam[1][a]
                for j in range(spec.harmonics[a]):
                    c = math.cos(lam[a][j])
                    sn = math.sin(lam[a][j])
                    sj = s[a][j]
                    sjs = sstar[a][j]
                    s[a][j] = sj * c + sjs * sn + g1 * d
                    sstar[a][j] = -sj * sn + sjs * c + g2 * d
        else:
            for a in range(len(spec.periods)):
                oldest = buf[a].pop(0)
                buf[a].append(oldest + gam[a] * d)
        lev = new_lev
        trend = new_trend
        if spec.p:
            dlag = [d] + dlag[:-1]
        if spec.q:
            elag = [eps] + elag[:-1]

    carry = {"level": lev, "trend": trend, "dlag": dlag, "elag": elag}
    if spec.trigonometric:
        carry["s"] = [list(v) for v in s]
        carry["sstar"] = [list(v) for v in sstar]
        carry["lam"] = lam
    else:
        carry["buf"] = [list(v) for v in buf]
    return resid, fitted, carry


def fit_seed_state(z, spec, theta, long_run_b=0.0):
    r"""Concentrate :math:`x_0` out of the likelihood by least squares.

    The residuals are affine in :math:`x_0`, so running the filter once
    with :math:`x_0 = 0` gives the intercept and once per basis vector
    (with :math:`z = 0`) gives the columns; the seed that minimises
    :math:`\sum \varepsilon_t^2` is then a linear solve, not a search.
    """
    n = len(z)
    ns = spec.n_states()
    zero = [0.0] * ns
    base, _, _ = bats_filter(z, spec, theta, zero, long_run_b)
    cols = []
    zeros_z = [0.0] * n
    for j in range(ns):
        e = [0.0] * ns
        e[j] = 1.0
        col, _, _ = bats_filter(zeros_z, spec, theta, e, long_run_b)
        cols.append(col)
    # residual(x0) = base + sum_j x0_j * col_j ; minimise its norm
    design = [[-cols[j][t] for j in range(ns)] for t in range(n)]
    sol = np.linalg.lstsq(np.asarray(design, dtype=float),
                          np.asarray(base, dtype=float), rcond=None)[0]
    return [float(sol[j]) for j in range(ns)]


# --------------------------------------------------------------------------
# the forecastability region
# --------------------------------------------------------------------------

def _flatten_carry(spec, carry):
    """The state vector in the same layout as ``x0``."""
    out = [carry["level"]]
    if spec.use_trend:
        out.append(carry["trend"])
    if spec.trigonometric:
        for a in range(len(spec.periods)):
            out += list(carry["s"][a])
            out += list(carry["sstar"][a])
    else:
        for a in range(len(spec.periods)):
            out += list(carry["buf"][a])
    out += list(carry["dlag"])
    out += list(carry["elag"])
    return out


def state_matrices(spec, theta):
    r"""Return :math:`(w, F, g)` of the innovations state space form.

    The measurement is :math:`y^{(\omega)}_t = w' x_{t-1} +
    \varepsilon_t` and the transition :math:`x_t = F x_{t-1} +
    g \varepsilon_t`. Both are read off the recursion itself rather
    than transcribed: the map is linear, so one filter step from each
    basis vector gives a column.
    """
    ns = spec.n_states()
    zero = [0.0] * ns
    w = []
    fcols = []
    for j in range(ns):
        e = [0.0] * ns
        e[j] = 1.0
        # z chosen so that eps = 0, which leaves x_1 = F e_j
        _, fit, carry = bats_filter([0.0], spec, theta, e)
        wj = fit[0]
        w.append(wj)
        _, _, carry = bats_filter([wj], spec, theta, e)
        fcols.append(_flatten_carry(spec, carry))
    # x0 = 0 and z = 1 gives eps = 1, so x_1 = g
    _, _, carry = bats_filter([1.0], spec, theta, zero)
    g = _flatten_carry(spec, carry)
    fmat = [[fcols[j][i] for j in range(ns)] for i in range(ns)]
    return w, fmat, g


def spectral_radius(spec, theta, tol=1e-6):
    r"""The largest :math:`|\lambda|` of :math:`D = F - g w'`, ignoring
    the structural unit root.

    Hyndman et al. (2007); the paper constrains estimation so that the
    characteristic roots of :math:`D` lie within the unit circle, which
    is what keeps a fitted model forecastable rather than merely a good
    in-sample smoother.

    One eigenvalue is always exactly 1 and must be excluded. The level
    and the seasonal indices are identified only up to a constant
    shift -- add :math:`c` to :math:`\ell` and subtract :math:`c` from
    every :math:`s^{(i)}` and nothing observable changes -- and that
    one-dimensional indeterminacy shows up as a unit eigenvalue of
    :math:`D` at *every* parameter value, the true one included.
    Testing the raw spectral radius therefore rejects everything.
    Excluding it still discriminates: on a seasonal series the true
    parameters leave the rest at 0.991 while a runaway
    :math:`\gamma = -0.5` pushes them to 1.040.
    """
    w, fmat, g = state_matrices(spec, theta)
    ns = len(w)
    d = [[fmat[i][j] - g[i] * w[j] for j in range(ns)] for i in range(ns)]
    ev = [abs(complex(v))
          for v in np.linalg.eigvals(np.asarray(d, dtype=float))]
    rest = [v for v in ev if abs(v - 1.0) >= tol]
    return max(rest) if rest else 0.0


def all_eigenvalues(spec, theta):
    """Every :math:`|\lambda|` of :math:`D`, largest first."""
    w, fmat, g = state_matrices(spec, theta)
    ns = len(w)
    d = [[fmat[i][j] - g[i] * w[j] for j in range(ns)] for i in range(ns)]
    return sorted((abs(complex(v)) for v
                   in np.linalg.eigvals(np.asarray(d, dtype=float))),
                  reverse=True)


def is_forecastable(spec, theta, tol=1e-8):
    """Whether ``theta`` lies inside the forecastability region."""
    return spectral_radius(spec, theta) < 1.0 - tol


def concentrated_loglik(y, resid, omega):
    r""":math:`-\frac{n}{2}\log \sum \varepsilon_t^2
    + (\omega - 1) \sum \log y_t`.

    The second term is the Jacobian of the Box-Cox transform, and it is
    what makes likelihoods at different :math:`\omega` comparable.
    """
    n = len(resid)
    sse = sum(v * v for v in resid)
    if sse <= 0.0:
        return float("inf")
    out = -0.5 * n * math.log(sse)
    if omega != 1.0:
        out += (omega - 1.0) * sum(math.log(float(v)) for v in y)
    return out


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def _bounds(spec):
    lo = [0.0]
    hi = [1.0]
    if spec.use_trend:
        lo.append(0.0)
        hi.append(1.0)
        if spec.damped:
            lo.append(0.8)
            hi.append(1.0)
    ns = 2 * len(spec.periods) if spec.trigonometric else len(spec.periods)
    # The paper does not put a box on gamma; what bounds it is the
    # forecastability region enforced in the objective below.
    lo += [-1.0] * ns
    hi += [1.0] * ns
    lo += [-0.99] * (spec.p + spec.q)
    hi += [0.99] * (spec.p + spec.q)
    if spec.use_box_cox:
        lo.append(0.0)
        hi.append(1.5)
    return lo, hi


def _starts(spec):
    """Several starting points for the smoothing parameters.

    Nelder-Mead builds its initial simplex by nudging each coordinate
    by 5% of its own value (scipy's rule, and the one in _sci_core), so
    a single start of alpha = 0.09 only ever explores a box 0.004
    wide. Starting from a small grid instead is what makes the search
    actually search.
    """
    out = []
    for a in (0.02, 0.1, 0.3):
        for gseed in (0.005, 0.05, 0.2):
            x = [a]
            if spec.use_trend:
                x.append(min(0.5 * a, 0.05))
                if spec.damped:
                    x.append(0.98)
            ns = (2 * len(spec.periods) if spec.trigonometric
                  else len(spec.periods))
            x += [gseed] * ns
            x += [0.0] * (spec.p + spec.q)
            if spec.use_box_cox:
                x.append(1.0)
            out.append(x)
    return out


def _fit_spec(y, spec, long_run_b=0.0, maxiter=2000):
    lo, hi = _bounds(spec)

    def clamp(th):
        return [min(max(th[j], lo[j]), hi[j]) for j in range(len(th))]

    def negll(th):
        th = clamp(list(th))
        omega = th[-1] if spec.use_box_cox else 1.0
        try:
            # "We can constrain the estimation to the forecastibility
            # region (Hyndman et al. 2007) so that the characteristic
            # roots of D lie within the unit circle." Without this the
            # optimiser buys in-sample fit with a seasonal state that
            # tracks the noise.
            rho = spectral_radius(spec, th)
            if rho >= 1.0:
                return 1e12 * (1.0 + rho)
            z = box_cox(y, omega)
            x0 = fit_seed_state(z, spec, th, long_run_b)
            resid, _, _ = bats_filter(z, spec, th, x0, long_run_b)
            ll = concentrated_loglik(y, resid, omega)
        except (ValueError, ZeroDivisionError, OverflowError):
            return 1e18
        if ll != ll or ll == float("inf"):
            return 1e18
        return -ll

    best_x = None
    best_f = float("inf")
    for st in _starts(spec):
        res = minimize(negll, st, method="nelder-mead", maxiter=maxiter)
        xr = list(res["x"] if isinstance(res, dict) else res.x)
        fr = negll(xr)
        if fr < best_f:
            best_f = fr
            best_x = xr
    th = clamp(best_x)
    omega = th[-1] if spec.use_box_cox else 1.0
    z = box_cox(y, omega)
    x0 = fit_seed_state(z, spec, th, long_run_b)
    resid, fitted, _ = bats_filter(z, spec, th, x0, long_run_b)
    ll = concentrated_loglik(y, resid, omega)
    k = spec.n_free() + spec.n_states()
    return {"theta": th, "x0": x0, "resid": resid, "fitted": fitted,
            "loglik": ll, "omega": omega,
            "aic": -2.0 * ll + 2.0 * k, "n_par": k}


def _forecast(spec, theta, x0, z, h, long_run_b=0.0):
    r"""Forecast by iterating the state equations with :math:`\varepsilon = 0`.

    Future innovations have mean zero, so the point forecast just runs
    equations 3b-3f forward from the final state. The MA terms die
    after :math:`q` steps; the AR recursion on :math:`d` continues.
    """
    h = int(h)
    if h < 0:
        raise ValueError("bats: the forecast horizon must not be negative")
    if h == 0:
        return []
    alpha, beta, phi, gam, ar, ma, _ = _unpack(spec, theta)
    _, _, c = bats_filter(z, spec, theta, x0, long_run_b)
    lev = c["level"]
    trend = c["trend"]
    dlag = list(c["dlag"])
    elag = list(c["elag"])
    if spec.trigonometric:
        s = [list(v) for v in c["s"]]
        sstar = [list(v) for v in c["sstar"]]
        lam = c["lam"]
    else:
        buf = [list(v) for v in c["buf"]]

    out = []
    for _ in range(h):
        if spec.trigonometric:
            seas = sum(sum(s[a]) for a in range(len(spec.periods)))
        else:
            seas = sum(buf[a][0] for a in range(len(spec.periods)))
        d = (sum(ar[j] * dlag[j] for j in range(spec.p))
             + sum(ma[j] * elag[j] for j in range(spec.q)))
        out.append(lev + phi * trend + seas + d)

        new_lev = lev + phi * trend + alpha * d
        new_trend = ((1.0 - phi) * long_run_b + phi * trend + beta * d
                     if spec.use_trend else 0.0)
        if spec.trigonometric:
            for a in range(len(spec.periods)):
                g1 = gam[0][a]
                g2 = gam[1][a]
                for j in range(spec.harmonics[a]):
                    cc = math.cos(lam[a][j])
                    sn = math.sin(lam[a][j])
                    sj = s[a][j]
                    sjs = sstar[a][j]
                    s[a][j] = sj * cc + sjs * sn + g1 * d
                    sstar[a][j] = -sj * sn + sjs * cc + g2 * d
        else:
            for a in range(len(spec.periods)):
                oldest = buf[a].pop(0)
                buf[a].append(oldest + gam[a] * d)
        lev = new_lev
        trend = new_trend
        if spec.p:
            dlag = [d] + dlag[:-1]
        if spec.q:
            elag = [0.0] + elag[:-1]     # future innovations are zero
    return out


def bats(y, seasonal_periods=(), use_box_cox=None, use_trend=None,
         damped=None, p=0, q=0, long_run_b=0.0, h=0, maxiter=2000):
    r"""Fit a BATS model, equations 3a-3f.

    ``use_box_cox``, ``use_trend`` and ``damped`` may each be ``True``,
    ``False``, or ``None`` to try both and keep the lower AIC -- which
    is the paper's own model selection.
    """
    y = [float(v) for v in y]
    if len(y) < 4:
        raise ValueError("bats: the series is too short")
    periods = [float(v) for v in seasonal_periods]
    if any(len(y) <= 2 * m for m in periods):
        raise ValueError("bats: the series is shorter than two full "
                         "cycles of a seasonal period")
    return _select(y, periods, None, use_box_cox, use_trend, damped,
                   p, q, long_run_b, h, maxiter)


def tbats(y, seasonal_periods=(), harmonics=None, use_box_cox=None,
          use_trend=None, damped=None, p=0, q=0, long_run_b=0.0, h=0,
          maxiter=2000):
    r"""Fit a TBATS model, equations 4a-4c for the seasonal part.

    ``harmonics`` defaults to the index-equivalent :math:`k_i`
    (:math:`m_i/2` even, :math:`(m_i-1)/2` odd). Fractional periods are
    allowed here and only here.
    """
    y = [float(v) for v in y]
    if len(y) < 4:
        raise ValueError("tbats: the series is too short")
    periods = [float(v) for v in seasonal_periods]
    if harmonics is None:
        harmonics = [len(seasonal_harmonics(m)) for m in periods]
    return _select(y, periods, list(harmonics), use_box_cox, use_trend,
                   damped, p, q, long_run_b, h, maxiter)


def _select(y, periods, harmonics, use_box_cox, use_trend, damped,
            p, q, long_run_b, h, maxiter):
    bc = [True, False] if use_box_cox is None else [bool(use_box_cox)]
    tr = [True, False] if use_trend is None else [bool(use_trend)]
    best = None
    tried = []
    for b in bc:
        for t in tr:
            dm = ([True, False] if (damped is None and t)
                  else [bool(damped) and t])
            for d in dm:
                spec = BatsSpec(periods, harmonics, b, t, d, p, q)
                fit = _fit_spec(y, spec, long_run_b, maxiter)
                tried.append((spec.label(), fit["aic"]))
                if best is None or fit["aic"] < best[1]["aic"]:
                    best = (spec, fit)
    spec, fit = best
    z = box_cox(y, fit["omega"])
    fc = _forecast(spec, fit["theta"], fit["x0"], z, int(h), long_run_b) \
        if h else []
    alpha, beta, phi, gam, ar, ma, omega = _unpack(spec, fit["theta"])
    return RichResult(payload={
        "model": spec.label(),
        "omega": omega,
        "phi": phi,
        "alpha": alpha,
        "beta": beta,
        "gamma": gam,
        "ar": ar,
        "ma": ma,
        "seed_state": fit["x0"],
        "fitted": inv_box_cox(fit["fitted"], omega),
        "fitted_transformed": fit["fitted"],
        "residuals": fit["resid"],
        "loglik": fit["loglik"],
        "aic": fit["aic"],
        "n_par": fit["n_par"],
        "sigma2": sum(v * v for v in fit["resid"]) / len(fit["resid"]),
        "spectral_radius": spectral_radius(spec, fit["theta"]),
        "forecastable": is_forecastable(spec, fit["theta"]),
        "forecast": inv_box_cox(fc, omega) if fc else [],
        "forecast_transformed": fc,
        "candidates": tried,
        "spec": spec,
        "method": ("De Livera, Hyndman & Snyder (2010) eq. 3a-3f"
                   + (" with the trigonometric seasonal of eq. 4a-4c"
                      if spec.trigonometric else "")),
        "note": ("the seed state is concentrated out by least squares, "
                 "not searched; the likelihood carries the Box-Cox "
                 "Jacobian (omega-1) sum log y_t so that fits at "
                 "different omega are comparable"),
    })


def cheatsheet():
    return ("bats: De Livera, Hyndman & Snyder (2010). BATS = Box-Cox "
            "transform, ARMA errors, Trend, Seasonal -- an innovations "
            "state space model with one seasonal index per period "
            "(eq. 3a-3f), a damped trend that converges to a long-run "
            "trend b rather than to zero, and ARMA(p, q) errors. TBATS "
            "swaps the seasonal index for a trigonometric one "
            "(eq. 4a-4c), which needs 2*sum(k_i) seeds instead of "
            "sum(m_i), handles NON-INTEGER periods such as 365.25/7, "
            "and gives deterministic seasonality when the smoothing "
            "parameters are zero. k_i = m_i/2 (even) or (m_i-1)/2 (odd) "
            "reproduces the index form exactly. The seed state is "
            "concentrated out by least squares.")
