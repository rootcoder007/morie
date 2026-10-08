# morie.fn -- function file (rootcoder007/morie)
"""Atmospheric dispersion: Pasquill-Gifford dispersion coefficients (Briggs 1973), Briggs plume
rise, the Gaussian plume and puff models with ground and mixing-lid reflection, an Eulerian
advection-diffusion solver and a Lagrangian random-walk particle model."""

from __future__ import annotations

import math
import warnings

from ._qpcore import ssum
from ._richresult import RichResult
from ._rng import random_normal

__all__ = [
    "pg_sigmas",
    "briggs_plume_rise",
    "gaussian_plume",
    "gaussian_puff",
    "advection_diffusion_2d",
    "lagrangian_particles",
]


def _num(x, what, *, minimum=None, positive=False, scalar=True, whole=False):
    """One finite number (or a list of them) for the dispersion inputs."""
    if isinstance(x, (bool, str)) or x is None:
        raise ValueError(f"`{what}` must be numeric")
    if scalar:
        if isinstance(x, (list, tuple)):
            raise ValueError(f"`{what}` must be a single number")
        vals = [x]
    else:
        vals = list(x) if isinstance(x, (list, tuple)) else [x]
    out = []
    for v in vals:
        if isinstance(v, (bool, str)):
            raise ValueError(f"`{what}` must be numeric")
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"`{what}` must be numeric") from None
        if not math.isfinite(f):
            raise ValueError(f"`{what}` must be finite")
        if positive and f <= 0:
            raise ValueError(f"`{what}` must be positive")
        if minimum is not None and f < minimum:
            raise ValueError(f"`{what}` must be at least {minimum}")
        if whole and f != math.floor(f):
            raise ValueError(f"`{what}` must be a whole number")
        out.append(f)
    return out[0] if scalar else out


def _lid(mixing_height, h):
    if mixing_height is None:
        return None
    lid = _num(mixing_height, "mixing_height", positive=True)
    if h > lid:
        warnings.warn(
            f"release height {h} is above the mixing height {lid}: the plume is above the lid "
            "and the reflection terms do not describe it",
            stacklevel=3,
        )
    return lid


# Briggs (1973) open-country and urban fits: (a_y, a_z, b_z, e_z); sigma_y = a_y x (1 + 1e-4 x)^-1/2
# (urban 4e-4); sigma_z = a_z x (1 + b_z x)^e_z
_RURAL = {
    "A": (0.22, 0.20, 0.0, 0.0),
    "B": (0.16, 0.12, 0.0, 0.0),
    "C": (0.11, 0.08, 0.0002, -0.5),
    "D": (0.08, 0.06, 0.0015, -0.5),
    "E": (0.06, 0.03, 0.0003, -1.0),
    "F": (0.04, 0.016, 0.0003, -1.0),
}
_URBAN = {
    "A": (0.32, 0.24, 0.001, 0.5),
    "B": (0.32, 0.24, 0.001, 0.5),
    "C": (0.22, 0.20, 0.0, 0.0),
    "D": (0.16, 0.14, 0.0003, -0.5),
    "E": (0.11, 0.08, 0.0015, -0.5),
    "F": (0.11, 0.08, 0.0015, -0.5),
}


def pg_sigmas(x, stability: str = "D", *, setting: str = "rural") -> RichResult:
    r"""Pasquill-Gifford horizontal and vertical dispersion coefficients (Briggs 1973 formulas).

    For downwind distance ``x`` (m) and stability class A-F: rural
    ``sigma_y = a_y x (1 + 0.0001 x)^(-1/2)``, urban
    ``sigma_y = a_y x (1 + 0.0004 x)^(-1/2)``, and
    ``sigma_z = a_z x (1 + b_z x)^(e_z)`` with Briggs' class coefficients
    (e.g. rural D: ``0.06 x (1 + 0.0015 x)^(-1/2)``).

    References
    ----------
    Briggs, G. A. (1973). Diffusion estimation for small emissions. ATDL
    Contribution 79, NOAA. Seinfeld, J. H. and Pandis, S. N. (2016).
    Atmospheric Chemistry and Physics, 3rd ed., Table 18.3.

    Examples
    --------
    >>> r = pg_sigmas([1000.0], "D")
    >>> round(r.sigma_y[0], 10), round(r.sigma_z[0], 10)
    (76.2770071396, 37.947331922)
    """
    tab = _RURAL if setting == "rural" else _URBAN
    ay, az, bz, ez = tab[stability.upper()]
    cy = 0.0001 if setting == "rural" else 0.0004
    xs = _num(x, "x", minimum=0.0, scalar=False)
    sy = [ay * v * (1.0 + cy * v) ** -0.5 for v in xs]
    sz = [az * v * (1.0 + bz * v) ** ez for v in xs]
    return RichResult(payload={"sigma_y": sy, "sigma_z": sz})


def briggs_plume_rise(
    x,
    u: float,
    *,
    diameter: float,
    exit_velocity: float,
    stack_temp: float,
    ambient_temp: float,
    stability: str = "D",
    dtheta_dz: float | None = None,
    g: float = 9.80616,
) -> RichResult:
    r"""Briggs buoyant plume rise (the EPA ISC3 formulation).

    Buoyancy flux ``F = g v_s d^2 (T_s - T_a) / (4 T_s)`` (m^4/s^3).
    Unstable and neutral classes (A-D): final rise
    ``21.425 F^(3/4) / u`` with ``x_f = 49 F^(5/8)`` when ``F < 55``, else
    ``38.71 F^(3/5) / u`` with ``x_f = 119 F^(2/5)``. Stable classes (E, F):
    ``s = g (dtheta/dz) / T_a`` (default lapse 0.020 K/m for E and 0.035 for F),
    final rise ``2.6 (F / (u s))^(1/3)`` reached at ``x_f = 2.0715 u / sqrt(s)``.
    Before ``x_f`` the transitional rise is ``1.6 F^(1/3) x^(2/3) / u``
    (capped at the final rise).

    References
    ----------
    Briggs, G. A. (1975). Plume rise predictions. In Lectures on Air Pollution
    and Environmental Impact Analyses, AMS, 59-111. U.S. EPA (1995). User's
    Guide for the Industrial Source Complex (ISC3) Dispersion Models, vol. II.

    Examples
    --------
    >>> r = briggs_plume_rise([100.0, 5000.0], 5.0, diameter=2.0, exit_velocity=15.0, stack_temp=400.0, ambient_temp=290.0)
    >>> round(r.flux, 10), [round(v, 8) for v in r.rise]
    (40.45041, [23.6659688, 68.72947432])
    """
    xs_in = _num(x, "x", minimum=0.0, scalar=False)
    u = _num(u, "u", positive=True)
    diameter = _num(diameter, "diameter", positive=True)
    exit_velocity = _num(exit_velocity, "exit_velocity", positive=True)
    stack_temp = _num(stack_temp, "stack_temp", positive=True)
    ambient_temp = _num(ambient_temp, "ambient_temp", positive=True)
    g = _num(g, "g", positive=True)
    if stack_temp <= ambient_temp:
        raise ValueError(
            "`stack_temp` must exceed `ambient_temp`: Briggs' formulas describe a buoyant plume, "
            "and a stack cooler than the air has no buoyancy flux"
        )
    if dtheta_dz is not None:
        dtheta_dz = _num(dtheta_dz, "dtheta_dz", positive=True)
    F = g * exit_velocity * diameter**2 * (stack_temp - ambient_temp) / (4.0 * stack_temp)
    cls = stability.upper()
    if cls in ("E", "F"):
        lapse = dtheta_dz if dtheta_dz is not None else (0.020 if cls == "E" else 0.035)
        s = g * lapse / ambient_temp
        final = 2.6 * (F / (u * s)) ** (1.0 / 3.0)
        xf = 2.0715 * u / math.sqrt(s)
    elif F < 55.0:
        final = 21.425 * F**0.75 / u
        xf = 49.0 * F ** (5.0 / 8.0)
    else:
        final = 38.71 * F**0.6 / u
        xf = 119.0 * F**0.4
    xs = xs_in
    rise = [final if v >= xf else min(1.6 * F ** (1.0 / 3.0) * v ** (2.0 / 3.0) / u, final) for v in xs]
    return RichResult(payload={"flux": F, "final_rise": final, "x_final": xf, "rise": rise})


def _vertical(z, h, sz, lid, n_images):
    t = math.exp(-((z - h) ** 2) / (2 * sz * sz)) + math.exp(-((z + h) ** 2) / (2 * sz * sz))
    if lid is not None:
        for j in range(1, n_images + 1):
            for sgn in (1, -1):
                off = 2 * sgn * j * lid
                t += math.exp(-((z - h + off) ** 2) / (2 * sz * sz)) + math.exp(-((z + h + off) ** 2) / (2 * sz * sz))
    return t


def gaussian_plume(
    q: float,
    u: float,
    h: float,
    receptors,
    *,
    stability: str = "D",
    setting: str = "rural",
    mixing_height: float | None = None,
    n_images: int = 3,
) -> list:
    r"""Steady-state Gaussian plume concentration with ground (and mixing-lid) reflection.

    ``C = Q / (2 pi u sigma_y sigma_z) exp(-y^2 / (2 sigma_y^2)) V`` with
    ``V = exp(-(z - H)^2 / (2 sigma_z^2)) + exp(-(z + H)^2 / (2 sigma_z^2))``
    plus, under a mixing lid at height ``L``, the image terms at
    ``z -+ H + 2 j L`` for ``j = +-1..n_images`` (Turner 1970). Receptors
    are ``(x, y, z)`` with ``x`` downwind (m); ``x <= 0`` gives 0. Sigmas come
    from :func:`pg_sigmas`; ``h`` is the effective stack height.

    References
    ----------
    Turner, D. B. (1970). Workbook of Atmospheric Dispersion Estimates. U.S.
    EPA AP-26. Seinfeld and Pandis (2016), eq. 18.60.

    Examples
    --------
    >>> round(gaussian_plume(100.0, 5.0, 50.0, [(1000.0, 0.0, 0.0)])[0], 12)
    0.000923237624
    """
    q = _num(q, "q", minimum=0.0)
    u = _num(u, "u", positive=True)
    h = _num(h, "h", minimum=0.0)
    n_images = int(_num(n_images, "n_images", minimum=0.0, whole=True))
    mixing_height = _lid(mixing_height, h)
    out = []
    for x, y, z in receptors:
        if x <= 0:
            out.append(0.0)
            continue
        s = pg_sigmas([x], stability, setting=setting)
        sy, sz = s.sigma_y[0], s.sigma_z[0]
        c = (
            q
            / (2 * math.pi * u * sy * sz)
            * math.exp(-(y * y) / (2 * sy * sy))
            * _vertical(z, h, sz, mixing_height, n_images)
        )
        out.append(c)
    return out


def gaussian_puff(
    mass: float,
    u: float,
    h: float,
    receptors,
    t: float,
    *,
    stability: str = "D",
    setting: str = "rural",
    mixing_height: float | None = None,
    n_images: int = 3,
) -> list:
    r"""Instantaneous Gaussian puff concentration at time ``t`` after release.

    ``C = M / ((2 pi)^(3/2) sigma_x sigma_y sigma_z) exp(-(x - u t)^2 / (2 sigma_x^2))
    exp(-y^2 / (2 sigma_y^2)) V`` with ``sigma_x = sigma_y`` evaluated at the
    travel distance ``u t`` (:func:`pg_sigmas`) and ``V`` the reflected
    vertical term of :func:`gaussian_plume`.

    References
    ----------
    Seinfeld and Pandis (2016), eq. 18.52. Zannetti, P. (1990). Air Pollution
    Modeling, ch. 7.

    Examples
    --------
    >>> round(gaussian_puff(1000.0, 5.0, 10.0, [(500.0, 0.0, 0.0)], 100.0)[0], 12)
    0.003334296408
    """
    mass = _num(mass, "mass", minimum=0.0)
    u = _num(u, "u", positive=True)
    h = _num(h, "h", minimum=0.0)
    t = _num(t, "t", positive=True)
    n_images = int(_num(n_images, "n_images", minimum=0.0, whole=True))
    mixing_height = _lid(mixing_height, h)
    s = pg_sigmas([u * t], stability, setting=setting)
    sy, sz = s.sigma_y[0], s.sigma_z[0]
    k = mass / ((2 * math.pi) ** 1.5 * sy * sy * sz)
    out = []
    for x, y, z in receptors:
        out.append(
            k
            * math.exp(-((x - u * t) ** 2) / (2 * sy * sy))
            * math.exp(-(y * y) / (2 * sy * sy))
            * _vertical(z, h, sz, mixing_height, n_images)
        )
    return out


def advection_diffusion_2d(
    c0, u: float, v: float, kx: float, ky: float, dx: float, dy: float, dt: float, n_steps: int, *, source=None
) -> RichResult:
    r"""Eulerian 2-D advection-diffusion by explicit finite differences.

    ``dC/dt = -u dC/dx - v dC/dy + K_x d2C/dx2 + K_y d2C/dy2 + S`` on the grid
    ``c0[i][j]`` (``i`` along ``x``), first-order upwind advection, central
    diffusion, forward Euler, zero-concentration boundaries. The explicit
    scheme is stable when ``|u| dt / dx + |v| dt / dy + 2 (K_x dt / dx^2 + K_y dt / dy^2) <= 1``
    (the von Neumann bound for upwind advection with forward-time
    centred-space diffusion), returned as ``stability_number`` with
    ``stable``. A call outside the bound is not run as given: each step is
    split into enough sub-steps of a smaller ``dt`` to satisfy it
    (``substeps``, ``dt``), with a warning, so the field is a solution over
    the same physical time rather than a numerical explosion. ``cfl`` and
    ``diffusion_number`` are reported for reference; each below 1 is
    necessary, not sufficient.

    References
    ----------
    Jacobson, M. Z. (2005). Fundamentals of Atmospheric Modeling, 2nd ed.,
    ch. 6. Seinfeld and Pandis (2016), ch. 18.

    Examples
    --------
    >>> c = [[0.0] * 5 for _ in range(5)]
    >>> c[2][2] = 1.0
    >>> r = advection_diffusion_2d(c, 0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 1.0, 1)
    >>> [round(a, 12) for a in r.field[2]]
    [0.0, 0.1, 0.6, 0.1, 0.0]
    >>> r.stability_number, r.stable, r.substeps
    (0.4, True, 1)
    """
    nx, ny = len(c0), len(c0[0])
    c = [[float(a) for a in row] for row in c0]
    for row in c:
        for a in row:
            if not math.isfinite(a):
                raise ValueError("`c0` must be a finite numeric grid")
    if source is not None and (len(source) != nx or any(len(row) != ny for row in source)):
        raise ValueError("`source` must have the dimensions of `c0`")
    u = _num(u, "u")
    v = _num(v, "v")
    kx = _num(kx, "kx", minimum=0.0)
    ky = _num(ky, "ky", minimum=0.0)
    dx = _num(dx, "dx", positive=True)
    dy = _num(dy, "dy", positive=True)
    dt = _num(dt, "dt", positive=True)
    n_steps = int(_num(n_steps, "n_steps", minimum=0.0, whole=True))
    ax, ay = u * dt / dx, v * dt / dy
    dxn, dyn = kx * dt / (dx * dx), ky * dt / (dy * dy)
    # the explicit upwind + FTCS scheme is stable when the advection and diffusion
    # numbers together stay within the unit interval; cfl < 1 alone is not enough
    stab = abs(ax) + abs(ay) + 2 * (dxn + dyn)
    substeps = 1
    if stab > 1:
        substeps = int(math.ceil(stab * (1 + 1e-9)))
        warnings.warn(
            "explicit scheme outside its stability bound (|u dt/dx| + |v dt/dy| + "
            f"2 (kx dt/dx^2 + ky dt/dy^2) = {stab:.3f} > 1): each step is taken as {substeps} "
            f"sub-steps of dt/{substeps}; pass a smaller dt to silence this",
            stacklevel=2,
        )
        dt = dt / substeps
        ax, ay = u * dt / dx, v * dt / dy
        dxn, dyn = kx * dt / (dx * dx), ky * dt / (dy * dy)
        stab = abs(ax) + abs(ay) + 2 * (dxn + dyn)
        n_steps = n_steps * substeps
    for _ in range(n_steps):
        new = [[0.0] * ny for _ in range(nx)]
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                adv_x = ax * (c[i][j] - c[i - 1][j]) if u >= 0 else ax * (c[i + 1][j] - c[i][j])
                adv_y = ay * (c[i][j] - c[i][j - 1]) if v >= 0 else ay * (c[i][j + 1] - c[i][j])
                dif = dxn * (c[i + 1][j] - 2 * c[i][j] + c[i - 1][j]) + dyn * (c[i][j + 1] - 2 * c[i][j] + c[i][j - 1])
                s = source[i][j] * dt if source is not None else 0.0
                new[i][j] = c[i][j] - adv_x - adv_y + dif + s
        c = new
    return RichResult(
        payload={
            "field": c,
            "mass": ssum(a for row in c for a in row) * dx * dy,
            "cfl": abs(ax) + abs(ay),
            "diffusion_number": 2 * (dxn + dyn),
            "stability_number": stab,
            "stable": stab <= 1,
            "dt": dt,
            "substeps": substeps,
        }
    )


def lagrangian_particles(
    n: int,
    x0: float,
    y0: float,
    u: float,
    v: float,
    kx: float,
    ky: float,
    dt: float,
    n_steps: int,
    *,
    seed: int = 0,
    grid=None,
) -> RichResult:
    r"""Lagrangian random-walk particle dispersion (homogeneous turbulence).

    Each of ``n`` particles moves as
    ``x_{k+1} = x_k + u dt + sqrt(2 K_x dt) xi``,
    ``y_{k+1} = y_k + v dt + sqrt(2 K_y dt) eta`` with standard normal
    ``xi, eta`` from the Philox stream (``seed``; step ``k`` uses streams
    ``2k`` and ``2k + 1``). With ``grid = (x_min, x_max, y_min, y_max, nx, ny)``
    the final particle counts per cell divided by ``n`` times cell area give
    a concentration field per unit released mass.

    References
    ----------
    Thomson, D. J. (1987). Criteria for the selection of stochastic models of
    particle trajectories in turbulent flows. J. Fluid Mech. 180, 529-556.

    Examples
    --------
    >>> r = lagrangian_particles(4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3)
    >>> r.x
    [3.0, 3.0, 3.0, 3.0]
    """
    if isinstance(n, bool) or int(n) != n or int(n) < 1:
        raise ValueError("`n` must be a positive number of particles")
    n = int(n)
    x0 = _num(x0, "x0")
    y0 = _num(y0, "y0")
    u = _num(u, "u")
    v = _num(v, "v")
    kx = _num(kx, "kx", minimum=0.0)
    ky = _num(ky, "ky", minimum=0.0)
    dt = _num(dt, "dt", positive=True)
    n_steps = int(_num(n_steps, "n_steps", minimum=0.0, whole=True))
    if grid is not None and len(grid) != 6:
        raise ValueError("`grid` must be (x_min, x_max, y_min, y_max, nx, ny)")
    xs, ys = [x0] * n, [y0] * n
    fx, fy = math.sqrt(2 * kx * dt), math.sqrt(2 * ky * dt)
    for k in range(n_steps):
        ex = random_normal(n, seed=seed, stream=2 * k) if n else []
        ey = random_normal(n, seed=seed, stream=2 * k + 1) if n else []
        xs = [xs[i] + u * dt + fx * float(ex[i]) for i in range(n)]
        ys = [ys[i] + v * dt + fy * float(ey[i]) for i in range(n)]
    out = {"x": xs, "y": ys, "mean_x": ssum(xs) / n, "mean_y": ssum(ys) / n}
    if grid is not None:
        x_min, x_max, y_min, y_max, gx, gy = grid
        wx, wy = (x_max - x_min) / gx, (y_max - y_min) / gy
        conc = [[0.0] * gy for _ in range(gx)]
        for a, b in zip(xs, ys):
            i, j = int(math.floor((a - x_min) / wx)), int(math.floor((b - y_min) / wy))
            if 0 <= i < gx and 0 <= j < gy:
                conc[i][j] += 1.0 / (n * wx * wy)
        out["concentration"] = conc
    return RichResult(payload=out)


def cheatsheet() -> str:
    return (
        "pg_sigmas / briggs_plume_rise / gaussian_plume / gaussian_puff / advection_diffusion_2d / "
        "lagrangian_particles -> atmospheric dispersion models."
    )
