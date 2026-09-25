# morie.fn -- function file (rootcoder007/morie)
"""Bandwidth selection via Sheather-Jones plug-in."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kbwpi"]


def _bw_bins(x, nb):
    """R's bw_den: counts of pairwise |i - j| bin distances on nb bins."""
    import math
    xmin = min(x)
    rang = (max(x) - xmin) * 1.01
    if rang <= 0:
        raise ValueError("data have zero range")
    dd = rang / nb
    # as the C code in R: (int)(x / dd), truncating toward zero, no shift
    idx = [int(v / dd) for v in x]
    cnt = [0] * nb
    n = len(x)
    for i in range(1, n):
        ii = idx[i]
        for j in range(i):
            cnt[abs(ii - idx[j])] += 1
    return dd, cnt


def _phi4(cnt, d, n, h):
    import math
    s = 0.0
    for i, c in enumerate(cnt):
        delta = (i * d / h) ** 2
        if delta >= 1000:
            break
        s += math.exp(-delta / 2) * (delta * delta - 6 * delta + 3) * c
    s = 2 * s + n * 3
    return s / (n * (n - 1) * h ** 5 * math.sqrt(2 * math.pi))


def _phi6(cnt, d, n, h):
    import math
    s = 0.0
    for i, c in enumerate(cnt):
        delta = (i * d / h) ** 2
        if delta >= 1000:
            break
        s += math.exp(-delta / 2) * (delta ** 3 - 15 * delta * delta + 45 * delta - 15) * c
    s = 2 * s - 15 * n
    return s / (n * (n - 1) * h ** 7 * math.sqrt(2 * math.pi))


def kbwpi(data: np.ndarray, method: str = "ste", nb: int = 1000,
          tol: float = 1e-12) -> dict:
    r"""
    Sheather-Jones plug-in bandwidth selector.

    The AMISE-optimal Gaussian-kernel bandwidth
    :math:`h = [R(K) / (n\,\mu_2(K)^2 \hat\psi_4(g))]^{1/5}` with the
    density functional :math:`\psi_4` estimated at a pilot bandwidth.
    ``method="ste"`` solves that equation with the pilot tied to h
    (Sheather and Jones's solve-the-equation rule); ``"dpi"`` is the
    two-stage direct plug-in. Both follow R's ``stats::bw.SJ`` step for
    step, pairwise distances binned on ``nb`` bins as there; the equation
    is solved to ``tol`` rather than bw.SJ's default of 10% of the
    bracket.

    The previous version used the physicists' Hermite polynomials (the
    Gaussian derivatives need the probabilists') with a sign fudge, so
    its functional could come out with the wrong sign and the bandwidth
    complex.

    Parameters
    ----------
    data : array-like
        1-d observations.
    method : {"ste", "dpi"}
    nb : int
        Number of bins for the pairwise distances (bw.SJ's nb).
    tol : float
        Root-finding tolerance for "ste".

    Returns
    -------
    dict
        ``bw_opt``, ``n``, ``method``.

    References
    ----------
    Sheather, S. J. & Jones, M. C. (1991). A reliable data-based bandwidth
        selection method for kernel density estimation. *JRSS-B*,
        53(3), 683-690. R Core Team, ``stats::bw.SJ``.
    """
    import math
    x = [float(v) for v in np.asarray(data, dtype=float).ravel().tolist()]
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 observations.")
    if method not in ("ste", "dpi"):
        raise ValueError("method must be 'ste' or 'dpi'")
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    q75, q25 = (float(v) for v in np.percentile(x, [75, 25]).tolist())
    scale = min(sd, (q75 - q25) / 1.349)
    if not scale > 0:
        scale = sd
    d, cnt = _bw_bins(x, int(nb))
    a = 1.24 * scale * n ** (-1 / 7)
    b = 1.23 * scale * n ** (-1 / 9)
    c1 = 1 / (2 * math.sqrt(math.pi) * n)
    TD = -_phi6(cnt, d, n, b)
    if not (TD > 0 and math.isfinite(TD)):
        raise ValueError("sample is too sparse to find TD")
    if method == "dpi":
        h = (c1 / _phi4(cnt, d, n, (2.394 / (n * TD)) ** (1 / 7))) ** 0.2
    else:
        alph2 = 1.357 * (_phi4(cnt, d, n, a) / TD) ** (1 / 7)

        def f(h):
            return (c1 / _phi4(cnt, d, n, alph2 * h ** (5 / 7))) ** 0.2 - h

        hmax = 1.144 * scale * n ** (-1 / 5)
        lo, hi = 0.1 * hmax, hmax
        for _ in range(99):
            if f(lo) * f(hi) <= 0:
                break
            if f(lo) < 0:
                lo *= 0.9
            else:
                hi *= 1.2
        flo = f(lo)
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if (fm < 0) == (flo < 0):
                lo, flo = mid, fm
            else:
                hi = mid
            if hi - lo < tol * max(1.0, abs(mid)):
                break
        h = 0.5 * (lo + hi)
    return RichResult(payload={"bw_opt": float(h), "n": n, "method": method})


def cheatsheet() -> str:
    return "kbwpi({data}) -> Sheather-Jones plug-in bandwidth."
