# morie.fn -- function file (rootcoder007/morie)
"""Wavelet denoising -- Donoho & Johnstone (1994), implemented natively.

Multi-level periodized DWT (Percival & Walden 2000: filters
periodized to N, an orthonormal transform, so reconstruction is
the exact transpose), soft/hard thresholding of the detail bands
at the universal threshold, then the inverse cascade. Daubechies
filter coefficients are the exact double-precision values from
PyWavelets' coefficient table (pywt/_extensions/c/
wavelets_coeffs.template.h); haar/db1 is 1/sqrt(2) analytically.
"""

from __future__ import annotations

import math

from . import _array_core as np
from ._richresult import RichResult, with_describe_pointer

__all__ = ["rangayyan_wavelet_denoise"]

_SQ2 = math.sqrt(2.0)

# Decomposition low-pass filters (pywt dec_lo ordering).
_DEC_LO = {
    "haar": [1.0 / _SQ2, 1.0 / _SQ2],
    "db1": [1.0 / _SQ2, 1.0 / _SQ2],
    "db2": [
        -1.294095225512603811744494188120241641745344506599652569070016e-01,
        2.241438680420133810259727622404003554678835181842717613871683e-01,
        8.365163037378079055752937809168732034593703883484392934953414e-01,
        4.829629131445341433748715998644486838169524195042022752011715e-01,
    ],
    "db3": [
        3.522629188570953660274066471551002932775838791743161039893406e-02,
        -8.544127388202666169281916918177331153619763898808662976351748e-02,
        -1.350110200102545886963899066993744805622198452237811919756862e-01,
        4.598775021184915700951519421476167208081101774314923066433867e-01,
        8.068915093110925764944936040887134905192973949948236181650920e-01,
        3.326705529500826159985115891390056300129233992450683597084705e-01,
    ],
    "db4": [
        -1.059740178506903210488320852402722918109996490637641983484974e-02,
        3.288301166688519973540751354924438866454194113754971259727278e-02,
        3.084138183556076362721936253495905017031482172003403341821219e-02,
        -1.870348117190930840795706727890814195845441743745800912057770e-01,
        -2.798376941685985421141374718007538541198732022449175284003358e-02,
        6.308807679298589078817163383006152202032229226771951174057473e-01,
        7.148465705529156470899219552739926037076084010993081758450110e-01,
        2.303778133088965008632911830440708500016152482483092977910968e-01,
    ],
}


def _filters(wavelet):
    try:
        lo = _DEC_LO[wavelet]
    except KeyError:
        raise ValueError(
            "unsupported wavelet %r; native families: %s"
            % (wavelet, sorted(_DEC_LO))) from None
    n = len(lo)
    # quadrature-mirror relation (Mallat 2009, A Wavelet Tour of
    # Signal Processing, 3rd ed., sec. 7.3): dec_hi[k] = (-1)^k
    # dec_lo[n-1-k]. The periodized transform is orthonormal, so the
    # synthesis operator is the transpose and uses the same filters.
    hi = [((-1) ** k) * lo[n - 1 - k] for k in range(n)]
    return lo, hi, lo, hi


def _dwt(x, lo, hi):
    """One periodized analysis level.

    ca[i] = sum_k lo[k] x[(2i+1-k) mod N] (Percival & Walden 2000,
    eq. 77b: filters periodized to N, odd-index subsampling). The rows
    of this map are the even circular shifts of lo and hi; for
    Daubechies filters they are orthonormal, so the transpose inverts
    it exactly. Odd-length input is extended by repeating the last
    sample (documented, only the final partial level is affected).
    """
    xs = list(x)
    if len(xs) % 2 == 1:
        xs.append(xs[-1])
    n = len(xs)
    f = len(lo)
    ca, cd = [], []
    for i in range(n // 2):
        a = d = 0.0
        for k in range(f):
            v = xs[(2 * i + 1 - k) % n]
            a += lo[k] * v
            d += hi[k] * v
        ca.append(a)
        cd.append(d)
    return ca, cd


def _idwt(ca, cd, lo, hi, n_out):
    """Transpose of :func:`_dwt` (exact inverse for orthonormal
    filters), cropped to ``n_out``."""
    n = 2 * len(ca)
    f = len(lo)
    out = [0.0] * n
    for i in range(len(ca)):
        for k in range(f):
            out[(2 * i + 1 - k) % n] += lo[k] * ca[i] + hi[k] * cd[i]
    return out[:n_out]


def _max_level(n, flen):
    """pywt.dwt_max_level: floor(log2(n / (flen - 1)))."""
    if flen <= 1 or n < flen:
        return 0
    return int(math.floor(math.log2(n / (flen - 1))))


def _threshold(d, T, mode):
    if mode == "hard":
        return [v if abs(v) > T else 0.0 for v in d]
    return [math.copysign(abs(v) - T, v) if abs(v) > T else 0.0
            for v in d]


def rangayyan_wavelet_denoise(x, wavelet="db4", level=None, mode="soft"):
    """Donoho-Johnstone wavelet denoising.

    Steps:

    1. Native DWT decomposition with ``wavelet`` to ``level`` levels.
    2. Estimate noise σ from the finest-scale detail coefficients via
       the median-absolute-deviation: σ = MAD(d1) / 0.6745.
    3. Universal threshold T = σ * sqrt(2 ln N).
    4. Apply ``soft`` (default) or ``hard`` thresholding to all detail
       coefficients (approximation untouched).
    5. Inverse DWT.

    Parameters
    ----------
    x : array-like
    wavelet : str
        Native wavelet name: haar, db1, db2, db3, db4 (default "db4").
    level : int, optional
        Decomposition depth. Defaults to the maximum useful level
        (the pywt.dwt_max_level rule).
    mode : {"soft", "hard"}
        Thresholding rule.

    Returns
    -------
    RichResult with keys ``signal`` (denoised), ``threshold``, ``sigma``,
    ``wavelet``, ``level``, ``mode``.

    References
    ----------
    Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by
        wavelet shrinkage. *Biometrika*, 81(3), 425-455. The universal
        threshold sigma*sqrt(2 log n) is theirs.
    Rangayyan, R. M., & Krishnan, S. (2024). *Biomedical Signal Analysis*
        (3rd ed.). Wiley-IEEE Press. Sec 8.14 "Application: Wavelet
        Denoising of PPG Signals", p.493.
    Mallat, S. (2009). *A Wavelet Tour of Signal Processing* (3rd ed.),
        sec. 7.3 (the analysis/synthesis filter bank).
    """
    xs = [float(v) for v in np.asarray(x, dtype=float).ravel()._flat()]
    n = len(xs)
    warnings_list: list[str] = []

    lo, hi, rec_lo, rec_hi = _filters(wavelet)
    max_level = _max_level(n, len(lo))
    if level is None:
        level = max_level
    if max_level == 0:
        warnings_list.append(
            "signal shorter than the filter; returned unchanged.")
        level = 0
    level = min(int(level), max_level) if max_level else 0

    cur = xs
    details = []
    lens = []
    for _ in range(level):
        lens.append(len(cur))
        cur, cd = _dwt(cur, lo, hi)
        details.append(cd)

    if details:
        d1 = details[0]                      # finest scale
        srt = sorted(abs(v) for v in d1)
        med = srt[len(srt) // 2]
        sigma = med / 0.6745
        T = sigma * math.sqrt(2.0 * math.log(max(n, 2)))
        details = [_threshold(d, T, mode) for d in details]
        y = cur
        for cd, n_out in zip(details[::-1], lens[::-1]):
            y = _idwt(y, cd, rec_lo, rec_hi, n_out)
        y = np.asarray(y[:n])
    else:
        sigma, T = 0.0, 0.0
        y = np.asarray(xs)

    res = RichResult(
        title="Wavelet denoising (Donoho-Johnstone)",
        summary_lines=[
            ("Wavelet", wavelet),
            ("Levels", level),
            ("σ (MAD/0.6745)", sigma),
            ("Universal threshold T", float(T)),
            ("Mode", mode),
        ],
        warnings=warnings_list,
        interpretation=(f"Denoised with {wavelet} at {level} levels, T={T:.4g}."),
        payload={
            "signal": y,
            "threshold": float(T),
            "sigma": sigma,
            "wavelet": wavelet,
            "level": level,
            "mode": mode,
        },
    )
    return with_describe_pointer(res, "rgwav")
