# morie.fn -- function file (rootcoder007/morie)
"""Spectral field tools: power-law and filtered fields, histogram transform, coherent field pairs."""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult
from ._rng import random_normal
from .sgsps import spectral_grf_sim

__all__ = ["power_law_field", "histogram_transform", "coherent_fields"]


def _freqs(n, d):
    return [(k if k <= (n - 1) // 2 else k - n) / (n * d) for k in range(n)]


def power_law_field(
    nx: int, ny: int, beta: float = 2.0, *, dx: float = 1.0, dy: float = 1.0, spectrum=None, seed: int = 1
) -> RichResult:
    r"""Gaussian field with a prescribed power spectrum by FFT filtering.

    White noise ``w`` on the ``nx x ny`` grid (Philox stream 0 of ``seed``,
    row-major) is transformed, multiplied by the amplitude filter
    ``H(k) = sqrt(P(|k|))`` and transformed back; the real part is the field,
    whose periodogram is ``|H|^2 |FFT(w)|^2``.  The default spectrum is the
    power law ``P(|k|) = |k|^-beta`` (``beta = 2`` a Brownian-like, ``beta
    -> 0`` a white field; Peitgen and Saupe 1988), with the zero frequency
    removed so the field has mean zero; any ``spectrum(k)`` of the radial
    frequency ``|k| = sqrt(kx^2 + ky^2)`` (cycles per unit) designs another
    filter.

    :param nx: Grid size along x.
    :param ny: Grid size along y.
    :param beta: Power-law exponent (ignored when ``spectrum`` is given).
    :param dx: Grid spacing along x.
    :param dy: Grid spacing along y.
    :param spectrum: Optional callable ``P(|k|)``.
    :param seed: Philox seed.
    :return: :class:`RichResult` with ``field`` (nx lists of ny values),
        ``filter`` (the amplitudes ``H``), ``kx``, ``ky``.

    References
    ----------
    Peitgen, H.-O. and Saupe, D. (eds) (1988). *The Science of Fractal
    Images*. Springer, New York.

    Examples
    --------
    >>> f = power_law_field(8, 8, beta=2.0, seed=3).field
    >>> abs(sum(sum(r) for r in f)) < 1e-12
    True
    """
    kx, ky = _freqs(nx, dx), _freqs(ny, dy)

    def P(k):
        if spectrum is not None:
            return float(spectrum(k))
        return 0.0 if k == 0.0 else k ** (-float(beta))

    H = [[math.sqrt(max(P(math.hypot(kx[i], ky[j])), 0.0)) for j in range(ny)] for i in range(nx)]
    w = [float(v) for v in random_normal(nx * ny, seed=seed, stream=0)]
    W = np.fft.fft2(np.array([[w[i * ny + j] for j in range(ny)] for i in range(nx)])).tolist()
    G = [[W[i][j] * H[i][j] for j in range(ny)] for i in range(nx)]
    g = np.fft.ifft2(np.array(G)).tolist()
    field = [[float(g[i][j].real) for j in range(ny)] for i in range(nx)]
    return RichResult(payload={"field": field, "filter": H, "kx": kx, "ky": ky})


def histogram_transform(values, target) -> RichResult:
    r"""Map values onto a target marginal distribution by ranks.

    The value of rank ``r`` (ties broken by position) among ``n`` becomes the
    type-7 sample quantile of ``target`` at probability ``(r - 1)/(n - 1)``;
    with ``len(target) == n`` this is exactly the sorted target placed in the
    rank order of the values.  Used to give a simulated Gaussian field the
    histogram of observed data (Journel and Deutsch 1993).

    Examples
    --------
    >>> histogram_transform([0.3, -1.2, 0.8, 0.1], [10, 40, 20, 30]).transformed
    [30.0, 10.0, 40.0, 20.0]
    """
    v = [float(a) for a in np.asarray(values, dtype=float).tolist()]
    t = sorted(float(a) for a in np.asarray(target, dtype=float).tolist())
    n, m = len(v), len(t)
    if n < 1 or m < 1:
        raise ValueError("values and target must be non-empty")
    order = sorted(range(n), key=lambda i: (v[i], i))
    out = [0.0] * n
    for r, i in enumerate(order):
        p = r / (n - 1) if n > 1 else 0.5
        h = (m - 1) * p
        lo = math.floor(h)
        out[i] = t[lo] + (h - lo) * (t[min(lo + 1, m - 1)] - t[lo])
    return RichResult(payload={"transformed": out, "ranks": [order.index(i) + 1 for i in range(n)]})


def coherent_fields(
    coords,
    cov_model: str = "exponential",
    cov_params: dict | None = None,
    *,
    coherence: float = 0.5,
    n_sims: int = 1,
    seed: int = 1,
) -> RichResult:
    r"""Two Gaussian fields with a constant spectral coherence.

    ``Z_1`` and an independent ``Z_2`` are simulated by
    :func:`~morie.fn.sgsps.spectral_grf_sim` (seeds ``seed`` and
    ``seed + 1``); ``Y = gamma Z_1 + sqrt(1 - gamma^2) Z_2`` then has the same
    covariance as ``Z_1`` and cross-covariance ``gamma C(h)``, i.e. coherence
    ``gamma`` at every frequency (the intrinsic coregionalisation model;
    Wackernagel 2003).

    Examples
    --------
    >>> g = [(x * 1.0, y * 1.0) for x in range(3) for y in range(3)]
    >>> r = coherent_fields(g, coherence=1.0, seed=2)
    >>> r.first == r.second
    True
    """
    gam = float(coherence)
    if not -1.0 <= gam <= 1.0:
        raise ValueError("coherence must lie in [-1, 1]")
    z1 = spectral_grf_sim(coords, cov_model, cov_params, n_sims=n_sims, seed=seed).extra["simulations"].tolist()
    z2 = spectral_grf_sim(coords, cov_model, cov_params, n_sims=n_sims, seed=seed + 1).extra["simulations"].tolist()
    s = math.sqrt(max(0.0, 1.0 - gam * gam))
    y = [[gam * a + s * b for a, b in zip(r1, r2)] for r1, r2 in zip(z1, z2)]
    return RichResult(
        payload={"first": z1[0] if n_sims == 1 else z1, "second": y[0] if n_sims == 1 else y, "coherence": gam}
    )


def cheatsheet() -> str:
    return "power_law_field, histogram_transform, coherent_fields: spectral field simulation tools."
