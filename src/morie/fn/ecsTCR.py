"""Equilibrium and transient climate sensitivity from a two-layer model.

Two numbers summarise how much a climate warms for a given forcing.
Equilibrium climate sensitivity is the warming once everything has
settled after CO2 doubles; transient climate response is the warming at
the moment of doubling in a run that raises CO2 by 1% a year, which
reaches doubling at year 70. ECS is a property of the balance, TCR of the
balance and the ocean's heat uptake together, so TCR is always the
smaller of the two.

The model is the two-layer energy balance that the assessment reports use
to emulate a general circulation model:

    C  dT/dt  = F(t) - lambda T - epsilon gamma (T - T_D)
    C_D dT_D/dt =                        gamma (T - T_D)

T is the surface layer, T_D the deep ocean, lambda the climate feedback
parameter, gamma the exchange coefficient, and epsilon the efficacy with
which deep-ocean uptake damps surface warming. Setting gamma to zero
collapses it to the one-layer model, which has a closed-form solution and
is used here as an anchor.

Forcing follows the logarithmic CO2 relation, F = F_2x log2(C/C_0), which
is what makes doubling the natural unit.

Three ways to get the sensitivities, because a study has whichever it has:

  parameters  lambda and gamma are known, so ECS = F_2x / lambda exactly
              and TCR comes from integrating the 1%/yr run to year 70.
  gregory     Only a step experiment is available. Regress the net
              top-of-atmosphere imbalance on surface temperature; the
              intercept is the forcing and minus the slope is lambda,
              which is Gregory et al.'s method. An abrupt-4xCO2 run is
              halved to reach the doubling value.
  emulate     Both a step run and a 1%/yr run are supplied and the
              parameters are fitted to them.

Three integrators, since the choice shows up in the third decimal of TCR:

  analytic    Diagonalise the 2x2 system. Exact for a constant forcing,
              and applied piecewise over the annual steps.
  rk4         Classical fourth-order Runge-Kutta.
  euler       Forward Euler, kept because it is what a quick calculation
              uses and it is useful to see how wrong it is.

Defaults are the AR6 values: F_2xCO2 = 3.93 W m-2. Charney's 1.5 to 4.5 K
range for ECS is reported alongside the estimate so a result can be placed
against the assessment that started the series.

References
  Charney, J.G. et al. (1979) "Carbon Dioxide and Climate: A Scientific
    Assessment." National Academy of Sciences, Washington DC. The 1.5 to
    4.5 K range.
  Forster, P. et al. (2021) "The Earth's Energy Budget, Climate Feedbacks,
    and Climate Sensitivity", Chapter 7 of Climate Change 2021: The
    Physical Science Basis, IPCC AR6 WG1, Cambridge University Press,
    923-1054, doi:10.1017/9781009157896.009. F_2xCO2 = 3.93 W m-2.
  Gregory, J.M. et al. (2004) "A new method for diagnosing radiative
    forcing and climate sensitivity", Geophysical Research Letters
    31(3), L03205, doi:10.1029/2003GL018747
  Held, I.M. et al. (2010) "Probing the fast and slow components of
    global warming by returning abruptly to preindustrial forcing",
    Journal of Climate 23(9), 2418-2427, doi:10.1175/2009JCLI3466.1
  Geoffroy, O. et al. (2013) "Transient climate response in a two-layer
    energy-balance model. Part I", Journal of Climate 26(6), 1841-1857,
    doi:10.1175/JCLI-D-12-00195.1
"""

import math

from ._richresult import RichResult

__all__ = ["ecs_tcr", "ecsTCR", "cheatsheet"]

_ROUTES = ("parameters", "gregory", "emulate")
_SOLVERS = ("analytic", "rk4", "euler")

# AR6 Chapter 7: the effective radiative forcing from doubling CO2.
F2X_AR6 = 3.93
# Charney (1979), the range that has anchored the question since.
CHARNEY_LOW = 1.5
CHARNEY_HIGH = 4.5


def _dot(a, b):
    """Compensated inner product, so both language arms agree bit for
    bit. Neither language's sum() is a plain double loop and they are not
    unfaithful in the same way."""
    s = 0.0
    c = 0.0
    for x, y in zip(a, b):
        t = x * y
        u = s + t
        if abs(s) >= abs(t):
            c += (s - u) + t
        else:
            c += (t - u) + s
        s = u
    return s + c


def _csum(v):
    s = 0.0
    c = 0.0
    for t in v:
        u = s + t
        if abs(s) >= abs(t):
            c += (s - u) + t
        else:
            c += (t - u) + s
        s = u
    return s + c


def _mean(v):
    return _csum(v) / len(v) if v else float("nan")


def _deriv(T, TD, F, lam, gam, eps, C, CD):
    dT = (F - lam * T - eps * gam * (T - TD)) / C
    dTD = gam * (T - TD) / CD
    return dT, dTD


def _step_analytic(T, TD, F, lam, gam, eps, C, CD, h):
    """One step of the exact solution for a forcing held constant over
    the step. The system is linear, so diagonalising the 2x2 matrix gives
    the answer outright; no tolerance, no step-size sensitivity."""
    if gam == 0.0:
        # The one-layer model, which is not a degenerate case to be
        # nursed through the 2x2 machinery but a closed form: the deep
        # ocean is decoupled and the surface relaxes exponentially
        # towards F/lambda. Going through the eigen-decomposition here
        # would divide by a zero determinant and silently fall back to
        # Runge-Kutta, costing six digits against a solution we know
        # exactly.
        eq = F / lam
        return eq + (T - eq) * math.exp(-lam * h / C), TD
    a11 = -(lam + eps * gam) / C
    a12 = eps * gam / C
    a21 = gam / CD
    a22 = -gam / CD
    tr = a11 + a22
    det = a11 * a22 - a12 * a21
    disc = tr * tr - 4.0 * det
    b1 = F / C
    b2 = 0.0
    if abs(det) < 1e-300 or disc < 0.0:
        return _step_rk4(T, TD, F, lam, gam, eps, C, CD, h)
    # equilibrium of the affine system, which the solution decays towards
    sd = math.sqrt(disc)
    r1 = 0.5 * (tr + sd)
    r2 = 0.5 * (tr - sd)
    eq1 = -(a22 * b1 - a12 * b2) / det
    eq2 = -(-a21 * b1 + a11 * b2) / det
    d1 = T - eq1
    d2 = TD - eq2
    if abs(r1 - r2) < 1e-14:
        return _step_rk4(T, TD, F, lam, gam, eps, C, CD, h)
    # e^{At} d in the eigenbasis of the 2x2 matrix
    v1a, v1b = (a12, r1 - a11) if abs(a12) > 1e-300 else (r1 - a22, a21)
    v2a, v2b = (a12, r2 - a11) if abs(a12) > 1e-300 else (r2 - a22, a21)
    dd = v1a * v2b - v2a * v1b
    if abs(dd) < 1e-300:
        return _step_rk4(T, TD, F, lam, gam, eps, C, CD, h)
    c1 = (d1 * v2b - d2 * v2a) / dd
    c2 = (d2 * v1a - d1 * v1b) / dd
    e1 = math.exp(r1 * h)
    e2 = math.exp(r2 * h)
    return (eq1 + c1 * v1a * e1 + c2 * v2a * e2,
            eq2 + c1 * v1b * e1 + c2 * v2b * e2)


def _step_rk4(T, TD, F, lam, gam, eps, C, CD, h):
    k1 = _deriv(T, TD, F, lam, gam, eps, C, CD)
    k2 = _deriv(T + 0.5 * h * k1[0], TD + 0.5 * h * k1[1], F, lam, gam,
                eps, C, CD)
    k3 = _deriv(T + 0.5 * h * k2[0], TD + 0.5 * h * k2[1], F, lam, gam,
                eps, C, CD)
    k4 = _deriv(T + h * k3[0], TD + h * k3[1], F, lam, gam, eps, C, CD)
    # The four-term combination goes through the compensated sum for the
    # same reason the dot products do: it is the one accumulation left in
    # the step, and a single differing bit here compounds over the run.
    return (T + h * _csum([k1[0], 2.0 * k2[0], 2.0 * k3[0], k4[0]]) / 6.0,
            TD + h * _csum([k1[1], 2.0 * k2[1], 2.0 * k3[1], k4[1]]) / 6.0)


def _step_euler(T, TD, F, lam, gam, eps, C, CD, h):
    d = _deriv(T, TD, F, lam, gam, eps, C, CD)
    return T + h * d[0], TD + h * d[1]


_STEPPERS = {"analytic": _step_analytic, "rk4": _step_rk4,
             "euler": _step_euler}


def integrate(forcing, lam, gamma=0.7, epsilon=1.0, C=8.0, C_deep=100.0,
              solver="analytic", dt=1.0, T0=0.0, TD0=0.0):
    """Run the two-layer model over a forcing series, one entry per year.

    Returns the surface series, the deep series and the net top-of-
    atmosphere imbalance, all of length len(forcing) + 1 for the state
    and len(forcing) for the imbalance.
    """
    if solver not in _SOLVERS:
        raise ValueError("ecsTCR: solver = %r; expected one of %s"
                         % (solver, ", ".join(_SOLVERS)))
    step = _STEPPERS[solver]
    T, TD = float(T0), float(TD0)
    Ts = [T]
    TDs = [TD]
    N = []
    for F in forcing:
        F = float(F)
        N.append(F - lam * T - (epsilon - 1.0) * gamma * (T - TD))
        T, TD = step(T, TD, F, lam, gamma, epsilon, C, C_deep, dt)
        Ts.append(T)
        TDs.append(TD)
    return Ts, TDs, N


def co2_forcing(ratio, f2x=F2X_AR6):
    """F = F_2x log2(C/C_0). Doubling gives exactly F_2x, which is what
    makes the doubling the natural unit for both sensitivities.

    The grouping matters. Written as f2x * log(r) / log(2) the division
    happens after the multiplication and a doubling comes back as
    3.9299999999999997 rather than 3.93 -- close enough for climate,
    not close enough for a definition. Dividing the logs first gives
    exactly 1 at a doubling and exactly 0 at no change.
    """
    return f2x * (math.log(ratio) / math.log(2.0))


def _ols(x, y):
    """Slope and intercept, by the centred formula."""
    n = len(x)
    mx = _mean(x)
    my = _mean(y)
    sxx = _csum([(v - mx) * (v - mx) for v in x])
    sxy = _csum([(x[i] - mx) * (y[i] - my) for i in range(n)])
    if sxx == 0.0:
        raise ValueError("ecsTCR: the temperature series has no spread, so "
                         "the Gregory regression is not identified")
    slope = sxy / sxx
    return slope, my - slope * mx


def ecs_tcr(model_run=None, CO2_traj=None, route="parameters",
            lam=None, gamma=0.7, epsilon=1.0, C=8.0, C_deep=100.0,
            f2x=F2X_AR6, solver="analytic", years=70, rate=0.01,
            temperature=None, imbalance=None, forcing_multiple=2.0,
            dt=1.0):
    """Equilibrium and transient climate sensitivity.

    Parameters
    ----------
    model_run : sequence, optional
        A surface temperature series, when one is being diagnosed. Alias
        for temperature, kept for the older call shape.
    CO2_traj : sequence, optional
        CO2 concentrations relative to pre-industrial, one per year. When
        absent the 1%/yr trajectory is built from rate and years.
    route : str
        parameters, gregory or emulate.
    lam : float
        Climate feedback parameter, W m-2 K-1. Required for the
        parameters route; fitted by the others.
    gamma, epsilon, C, C_deep : float
        Two-layer parameters: exchange coefficient, deep-uptake efficacy,
        and the two heat capacities in W yr m-2 K-1.
    f2x : float
        Forcing from doubling CO2. Defaults to AR6's 3.93 W m-2.
    solver : str
        analytic, rk4 or euler.
    years : int
        Length of the 1%/yr run. 70 is the year CO2 doubles.
    rate : float
        Annual fractional CO2 increase. 0.01 is the standard experiment.
    imbalance : sequence, optional
        Net top-of-atmosphere imbalance, for the Gregory route.
    forcing_multiple : float
        What the step experiment did: 2 for abrupt-2x, 4 for abrupt-4x.
        The Gregory estimate is scaled to the doubling value by log2 of
        this.

    Returns
    -------
    RichResult
        ecs, tcr, tcr_ecs_ratio, lambda, f2x, realised_warming_fraction,
        temperature, deep_temperature, imbalance, charney_range,
        within_charney, route, solver, method.
    """
    if route not in _ROUTES:
        raise ValueError("ecsTCR: route = %r; expected one of %s"
                         % (route, ", ".join(_ROUTES)))
    if temperature is None:
        temperature = model_run

    fitted = None
    if route in ("gregory", "emulate"):
        if temperature is None or imbalance is None:
            raise ValueError("ecsTCR: the %s route needs both a temperature "
                             "series and the net imbalance" % route)
        T = [float(v) for v in temperature]
        N = [float(v) for v in imbalance]
        if len(T) != len(N):
            raise ValueError("ecsTCR: temperature has %d entries and "
                             "imbalance %d" % (len(T), len(N)))
        slope, intercept = _ols(T, N)
        lam_fit = -slope
        if lam_fit <= 0.0:
            raise ValueError("ecsTCR: the regression gives a non-positive "
                             "feedback parameter (%g), so the system has no "
                             "equilibrium" % lam_fit)
        # The intercept is the forcing of whatever step was run; scale to
        # the doubling value by the logarithmic CO2 relation.
        scale = math.log(forcing_multiple) / math.log(2.0)
        f2x = intercept / scale
        lam = lam_fit
        fitted = {"slope": slope, "intercept": intercept,
                  "forcing_multiple": forcing_multiple}

    if lam is None:
        raise ValueError("ecsTCR: give lam, or use a route that fits it")
    if lam <= 0.0:
        raise ValueError("ecsTCR: lam = %g; a non-positive feedback "
                         "parameter has no equilibrium" % lam)

    ecs = f2x / lam

    if CO2_traj is None:
        # Built by repeated multiplication, not by raising to a power.
        # R's ^ special-cases an integer exponent into repeated squaring
        # while Python's ** calls libm pow(), so the two disagree in the
        # last bits for the same nominal trajectory -- and the whole run
        # is downstream of it.
        traj = []
        acc = 1.0
        for _ in range(int(years)):
            acc = acc * (1.0 + rate)
            traj.append(acc)
    else:
        traj = [float(v) for v in CO2_traj]
    forcing = [co2_forcing(r, f2x) for r in traj]
    Ts, TDs, N = integrate(forcing, lam, gamma, epsilon, C, C_deep,
                           solver=solver, dt=dt)

    # TCR is the warming at the moment of doubling. With the standard
    # 1%/yr trajectory that is year 70; with a supplied trajectory, find
    # the first year the concentration reaches twice pre-industrial.
    idx = len(traj) - 1
    for i, r in enumerate(traj):
        if r >= 2.0:
            idx = i
            break
    tcr = Ts[idx + 1]

    return RichResult(payload={
        "ecs": ecs,
        "tcr": tcr,
        "tcr_ecs_ratio": tcr / ecs if ecs != 0.0 else float("nan"),
        "lambda": lam,
        "f2x": f2x,
        "doubling_year": idx + 1,
        "realised_warming_fraction": tcr / ecs if ecs != 0.0 else float("nan"),
        "temperature": Ts,
        "deep_temperature": TDs,
        "imbalance": N,
        "fitted": fitted,
        "charney_range": [CHARNEY_LOW, CHARNEY_HIGH],
        "within_charney": bool(CHARNEY_LOW <= ecs <= CHARNEY_HIGH),
        "route": route,
        "solver": solver,
        "method": ("two-layer energy balance (Held et al. 2010; Geoffroy "
                   "et al. 2013), ECS = F_2x / lambda, TCR at CO2 "
                   "doubling in a %g%%/yr run, %s route, %s solver"
                   % (rate * 100.0, route, solver)),
    })


ecsTCR = ecs_tcr


def cheatsheet():
    return ("ecsTCR: ECS and TCR from a two-layer energy balance. "
            "route = parameters | gregory | emulate; "
            "solver = analytic | rk4 | euler. F_2xCO2 defaults to AR6's "
            "3.93 W m-2; Charney (1979) put ECS at 1.5-4.5 K.")
