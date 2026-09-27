# morie.fn -- function file (rootcoder007/morie)
"""Neyman-Scott cluster processes: Thomas, Cauchy and Matern kernels."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult
from ._rng import normal_quantile, random_uniform


def _matclust_h(z):
    if z >= 1:
        return 1.0
    return 2 + (1 / math.pi) * (
        (8 * z * z - 4) * math.acos(z)
        - 2 * math.asin(z)
        + 4 * z * math.sqrt((1 - z * z) ** 3)
        - 6 * z * math.sqrt(1 - z * z)
    )


def _matclust_g(z):
    if z >= 1:
        return 0.0
    return (2 / math.pi) * (math.acos(z) - z * math.sqrt(1 - z * z))


def _chunks(mean):
    return max(1, math.ceil(mean / 500.0))


def _poisson(mean, us):
    """Inversion on ``len(us)`` independent parts of mean at most 500 (exp(-mean) stays representable)."""
    part = mean / len(us)
    total = 0
    for u in us:
        k, p = 0, math.exp(-part)
        c = p
        while u > c and p > 0:
            k += 1
            p *= part / k
            c += p
        total += k
    return total


def neyman_scott_process(
    kappa, mu, scale, *, kernel: str = "thomas", r=None, window=None, simulate: int = 0, seed: int = 0
) -> DescriptiveResult:
    """Neyman-Scott cluster process: moments and simulation.

    Parents form a Poisson process of intensity ``kappa``; each has a
    Poisson(``mu``) number of offspring displaced by the kernel: ``thomas``
    (isotropic Gaussian, standard deviation ``scale``), ``cauchy``
    (bivariate Cauchy with scale ``scale``, density ``(1 + |x|^2 /
    scale^2)^(-3/2) / (2 pi scale^2)``) or ``matern`` (uniform in a disc of
    radius ``scale``). The intensity is ``kappa mu`` and

    - Thomas: ``K(r) = pi r^2 + (1 - exp(-r^2 / (4 scale^2))) / kappa``,
      ``g(r) = 1 + exp(-r^2 / (4 scale^2)) / (4 pi kappa scale^2)``;
    - Cauchy (``eta^2 = 4 scale^2``): ``K(r) = pi r^2 + (1 - (1 +
      r^2/eta^2)^(-1/2)) / kappa``, ``g(r) = 1 + (1 + r^2/eta^2)^(-3/2) /
      (2 pi eta^2 kappa)``;
    - Matern: ``K(r) = pi r^2 + h(r / (2 scale)) / kappa`` and
      ``g(r) = 1 + g0(r / (2 scale)) / (pi kappa scale^2)`` with the
      disc-overlap functions of Matern (1960).

    These are the ``K`` and ``pcf`` entries of
    ``spatstat.random::spatstatClusterModelInfo``. With ``simulate > 0``
    and a rectangular ``window`` the process is simulated with parents in
    the window dilated by the radius beyond which an offspring falls with
    probability ``1e-3`` (Thomas, Matern exact) or ``1e-2`` (Cauchy).

    :param kappa: Parent intensity.
    :param mu: Mean number of offspring per parent.
    :param scale: Kernel scale.
    :param kernel: ``"thomas"``, ``"cauchy"`` or ``"matern"``.
    :param r: Distances at which to evaluate ``K`` and ``g``.
    :param window: Rectangle ``(xmin, xmax, ymin, ymax)`` for simulation.
    :param simulate: Number of patterns to simulate.
    :param seed: Philox key (pattern ``s`` uses streams ``4s`` to ``4s+3``).
    :return: DescriptiveResult; ``value`` is the intensity; ``extra`` has
        ``K``, ``pcf``, ``r`` and ``simulated``.

    References
    ----------
    Neyman, J. and Scott, E. L. (1958). Statistical approach to problems of
    cosmology. Journal of the Royal Statistical Society B 20, 1-43.

    Thomas, M. (1949). A generalization of Poisson's binomial limit for use
    in ecology. Biometrika 36, 18-25.

    Ghorbani, M. (2013). Cauchy cluster process. Metrika 76, 697-706.

    Examples
    --------
    >>> r = neyman_scott_process(10.0, 5.0, 0.05, r=[0.1])
    >>> r.value, round(r.extra["K"][0], 12)
    (50.0, 0.094627982419)
    """
    kappa, mu, s = float(kappa), float(mu), float(scale)
    rs = [] if r is None else [float(v) for v in r]
    if kernel == "thomas":
        K = [math.pi * t * t + (1 - math.exp(-t * t / (4 * s * s))) / kappa for t in rs]
        g = [1 + math.exp(-t * t / (4 * s * s)) / (4 * math.pi * kappa * s * s) for t in rs]
        reach = s * math.sqrt(2 * math.log(1000.0))
    elif kernel == "cauchy":
        e2 = 4 * s * s
        K = [math.pi * t * t + (1 - 1 / math.sqrt(1 + t * t / e2)) / kappa for t in rs]
        g = [1 + (1 + t * t / e2) ** -1.5 / (2 * math.pi * e2 * kappa) for t in rs]
        reach = s * math.sqrt(1e4 - 1)
    elif kernel == "matern":
        K = [math.pi * t * t + _matclust_h(t / (2 * s)) / kappa for t in rs]
        g = [1 + _matclust_g(t / (2 * s)) / (math.pi * kappa * s * s) for t in rs]
        reach = s
    else:
        raise ValueError("kernel must be 'thomas', 'cauchy' or 'matern'")
    sims = []
    if simulate > 0:
        if window is None:
            raise ValueError("simulation needs a window")
        x0, x1, y0, y1 = (float(v) for v in window)
        X0, X1, Y0, Y1 = x0 - reach, x1 + reach, y0 - reach, y1 + reach
        lam_p = kappa * (X1 - X0) * (Y1 - Y0)
        cm = _chunks(mu)
        for k in range(simulate):
            npar = _poisson(lam_p, [float(v) for v in random_uniform(_chunks(lam_p), seed=seed, stream=4 * k)])
            pu = [float(v) for v in random_uniform((2 + cm) * npar, seed=seed, stream=4 * k + 1)] if npar else []
            counts = [_poisson(mu, pu[(2 + cm) * j + 2 : (2 + cm) * (j + 1)]) for j in range(npar)]
            tot = sum(counts)
            q = [float(v) for v in normal_quantile(random_uniform(3 * tot, seed=seed, stream=4 * k + 2))] if tot else []
            v = [float(t) for t in random_uniform(2 * tot, seed=seed, stream=4 * k + 3)] if tot else []
            pat, c = [], 0
            for j in range(npar):
                px, py = X0 + pu[(2 + cm) * j] * (X1 - X0), Y0 + pu[(2 + cm) * j + 1] * (Y1 - Y0)
                for _ in range(counts[j]):
                    if kernel == "thomas":
                        dx, dy = s * q[3 * c], s * q[3 * c + 1]
                    elif kernel == "cauchy":
                        dx, dy = s * q[3 * c] / abs(q[3 * c + 2]), s * q[3 * c + 1] / abs(q[3 * c + 2])
                    else:
                        rad, ang = s * math.sqrt(v[2 * c]), 2 * math.pi * v[2 * c + 1]
                        dx, dy = rad * math.cos(ang), rad * math.sin(ang)
                    c += 1
                    x, y = px + dx, py + dy
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        pat.append([x, y])
            sims.append(pat)
    return DescriptiveResult(
        name="neyman_scott_process", value=kappa * mu, extra={"K": K, "pcf": g, "r": rs, "simulated": sims}
    )


nsproc = neyman_scott_process


def cheatsheet() -> str:
    return "neyman_scott_process(kappa, mu, scale) -> Thomas / Cauchy / Matern cluster process moments and simulation"
