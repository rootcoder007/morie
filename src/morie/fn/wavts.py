# morie.fn -- function file (rootcoder007/morie)
"""Discrete wavelet decomposition for time series (Percival & Walden 2000)."""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["wavelet_time_series"]


# Percival & Walden scaling filters {g_l}, in the book's own orientation.
#
# haar is eq (75c) and D(4) is eq (75d), p.75:
#     g0 = (1+sqrt3)/(4 sqrt2), g1 = (3+sqrt3)/(4 sqrt2),
#     g2 = (3-sqrt3)/(4 sqrt2), g3 = (1-sqrt3)/(4 sqrt2)
# LA(8) is P&W's least-asymmetric filter, p.107 ("also called 'symmlets'").
# P&W print the LA filters only as figures (Figures 108a/108b) -- the numeric
# coefficients live on the book's web site -- so these were cross-checked
# against the CRAN `wavelets` package, which implements P&W's conventions:
# `wt.filter("la8")@g` agrees to 9 decimals, as does its @h with the QMF below.
#
# The orientation is NOT recoverable from PyWavelets. pywt stores db2's dec_lo
# as the reverse of P&W's g, but sym4's dec_lo in the SAME order as P&W's g, so
# a uniform `dec_lo[::-1]` mapping silently time-reverses LA(8).
_PW_SCALING = {
    "haar": (0.7071067811865475, 0.7071067811865475),
    "d4": (
        0.4829629131445341,
        0.8365163037378079,
        0.2241438680420134,
        -0.1294095225512604,
    ),
    "la8": (
        -0.07576571478935668,
        -0.02963552764596039,
        0.49761866763256290,
        0.80373875180538600,
        0.29785779560560505,
        -0.09921954357695636,
        -0.01260396726226383,
        0.03222310060407815,
    ),
}


def _pw_filters(name):
    """Scaling and wavelet filters, P&W eq (75b) p.75: h_l = (-1)^l g_{L-1-l}."""
    g = np.asarray(_PW_SCALING[name], dtype=float)
    h = g[::-1] * (-1.0) ** np.arange(g.size)
    return g, h


def _pw_dwt(y, name, level):
    """P&W pyramid algorithm, eq (77b) p.77, periodic ('circular') boundary.

        V_{1,t} = sum_{l=0}^{L-1} g_l X_{2t+1-l mod N},  t = 0, ..., N/2 - 1

    and identically for W with {h_l}. Returns ``(V_J, [W_1, ..., W_J])``.
    """
    g, h = _pw_filters(name)
    v = np.asarray(y, dtype=float)
    Ws = []
    for _ in range(level):
        N = v.size
        half = N // 2
        t = np.arange(half)
        W = np.zeros(half)
        V = np.zeros(half)
        for ell in range(g.size):
            idx = (2 * t + 1 - ell) % N
            W += h[ell] * v[idx]
            V += g[ell] * v[idx]
        Ws.append(W)
        v = V
    return v, Ws


def wavelet_time_series(x, wavelet="haar", level=None):
    r"""Multiresolution decomposition via the discrete wavelet transform.

    .. math::

        x(t) = \sum_j \sum_k c_{j,k}\,\psi_{j,k}(t)

    Parameters
    ----------
    x : array-like
        Univariate time series.
    wavelet : str, default ``"haar"``
        Wavelet family (passed to ``pywt`` when available).
    level : int, optional
        Decomposition depth; defaults to ``floor(log2 n)``.

    Returns
    -------
    RichResult
        keys: ``approximation`` (cA at the deepest level), ``details``
        (list of cD coefficients level-by-level), ``energies`` (variance
        per level), ``level``, ``n``, ``method``.

    Raises
    ------
    ValueError
        If ``x`` has fewer than 4 observations, or if a non-Haar wavelet is
        requested without PyWavelets installed.

    References
    ----------
    Percival, D. B., & Walden, A. T. (2000). *Wavelet Methods for Time
        Series Analysis*. Cambridge University Press. DWT p.56; jth level
        wavelet detail :math:`\mathcal{D}_j` p.64; multiresolution analysis
        p.65; energy (squared norm) :math:`\mathcal{E}_X` pp.42, 72.
    PyWavelets developers. ``pywt.wavedec`` reference documentation.
        https://pywavelets.readthedocs.io/en/latest/ref/dwt-discrete-wavelet-transform.html

    Notes
    -----
    The orthonormal DWT preserves energy, so the returned ``energies`` sum to
    :math:`\sum_t x_t^2` (P&W's :math:`\mathcal{E}_X`, pp.42, 72). That is
    the identity the tests pin, and it is what would break first if the
    filter normalisation drifted.

    ``energies`` is aligned with the returned arrays: ``energies[0]`` is the
    approximation and ``energies[i + 1]`` is ``details[i]``. It previously was
    not -- ``details`` was ordered deepest-first while ``energies`` ran
    shallowest-first, so the two disagreed for every level beyond the first.

    PyWavelets is an optional extra (``pip install morie[wavelets]``). Without
    it only ``wavelet="haar"`` is available, via a pure-NumPy implementation;
    any other family raises rather than silently returning Haar. The previous
    code caught every exception from the pywt path and fell through, so a
    request for ``db4`` -- or a typo -- returned Haar coefficients with no
    warning.
    """
    y = np.asarray(x, dtype=float).ravel()
    n = y.size
    if n < 4:
        raise ValueError(f"Need at least 4 observations, got {n}.")
    max_level = int(np.floor(np.log2(n)))
    if level is None:
        level = min(max(max_level, 1), 6)
    level = int(min(level, max_level))

    key = str(wavelet).lower()
    if key in _PW_SCALING:
        # Section 4.4's DWT is defined for N = 2^J, so pad to the next power of
        # two. The padding is zeros, which leaves the energy identity intact.
        # This mirrors the R implementation exactly.
        N = 1 << int(np.ceil(np.log2(n)))
        yp = np.concatenate([y, np.zeros(N - n)]) if N > n else y
        J = min(level, int(np.floor(np.log2(N))))
        cA, Ws = _pw_dwt(yp, key, J)
        # Ws is shallowest-first (W_1 ... W_J); reverse so details run
        # deepest-first and energies[i + 1] is the energy of details[i].
        details = Ws[::-1]
        energies = [float(np.sum(cA**2))] + [float(np.sum(c**2)) for c in details]
        return RichResult(
            payload={
                "approximation": cA,
                "details": details,
                "energies": energies,
                "level": int(J),
                "n": int(n),
                "wavelet": key,
                "method": (
                    f"DWT (Percival & Walden pyramid, wavelet={key}, level={J}, "
                    "periodic boundary, odd-index subsampling per eq (77b) p.77)"
                ),
            }
        )

    try:
        import pywt
    except ImportError:
        pywt = None

    if pywt is None and str(wavelet).lower() not in ("haar", "db1"):
        raise ValueError(
            f"wavelet={wavelet!r} needs PyWavelets, which is not installed. "
            "Install it with `pip install morie[wavelets]`, or use "
            'wavelet="haar", which is implemented natively. Returning Haar '
            "for a non-Haar request would be a silently wrong basis."
        )

    if pywt is not None:
        # Errors from pywt propagate: an unknown wavelet name is a caller
        # mistake, not a reason to substitute a different basis.
        # mode="periodization", not pywt's default "symmetric".
        #
        # P&W define the DWT with filters "periodized to N" (Conventions and
        # Notation, A_j / B_j), which makes the transform orthonormal and
        # energy-preserving. pywt's default symmetric extension is redundant:
        # it returns more coefficients than input samples, and the energies do
        # NOT sum to sum(x^2). Measured on 64 Gaussian samples with db4 at
        # level 3, energy preservation failed under "symmetric" and holds
        # under "periodization" -- and only the latter agrees with the native
        # Haar path, so swapping PyWavelets in or out no longer changes the
        # answer.
        coeffs = pywt.wavedec(y, wavelet, level=level, mode="periodization")
        cA = coeffs[0]
        cDs = coeffs[1:]
        # coeffs is [cA_n, cD_n, ..., cD_1], so energies line up with
        # [approximation] + details as returned.
        energies = [float(np.sum(c**2)) for c in coeffs]
        return RichResult(
            payload={
                "approximation": np.asarray(cA),
                "details": [np.asarray(c) for c in cDs],
                "energies": energies,
                "level": int(level),
                "n": int(n),
                "wavelet": wavelet,
                "method": f"DWT via pywt (wavelet={wavelet}, level={level}, mode=periodization)",
            }
        )

    # Pure-NumPy Haar DWT (used when PyWavelets is absent).
    cA = y.copy()
    cDs = []
    for _ in range(level):
        if cA.size < 2:
            break
        if cA.size % 2 == 1:
            cA = np.concatenate([cA, cA[-1:]])
        even = cA[0::2]
        odd = cA[1::2]
        cA_new = (even + odd) / np.sqrt(2.0)
        cD = (even - odd) / np.sqrt(2.0)
        cDs.append(cD)
        cA = cA_new
    # cDs is built shallowest-first (cD_1 ... cD_n); reverse to match pywt's
    # deepest-first convention, and build energies from the SAME ordered list
    # so energies[i + 1] is always the energy of details[i].
    details = cDs[::-1]
    energies = [float(np.sum(cA**2))] + [float(np.sum(c**2)) for c in details]
    return RichResult(
        payload={
            "approximation": cA,
            "details": details,
            "energies": energies,
            "level": int(level),
            "n": int(n),
            "wavelet": "haar",
            "method": "Haar DWT (numpy fallback)",
        }
    )


def cheatsheet():
    return "wavts: Wavelet decomposition (Percival & Walden 2000)."
