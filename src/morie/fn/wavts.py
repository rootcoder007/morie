# morie.fn -- function file (rootcoder007/morie)
"""Discrete wavelet decomposition for time series (Percival & Walden 2000)."""

from __future__ import annotations

import numpy as np

from ._richresult import RichResult

__all__ = ["wavelet_time_series"]


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
