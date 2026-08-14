# morie.fn -- wave3 coordinator batch (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Planck blackbody spectrum, wavelength form."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["plncF", "planck_function"]


def planck_function(lam, T, h=6.62607015e-34, c=299792458.0,
                    kB=1.380649e-23):
    """
    Planck spectral radiance in the WAVELENGTH form,

        B(lam, T) = (2 h c^2 / lam^5) / (exp(h c / (lam kB T)) - 1)

    (W sr-1 m-3). The sibling module `plank` implements the FREQUENCY
    form B(nu, T); the two are related exactly by the Jacobian
    B_lam(lam) = B_nu(c/lam) c / lam^2, which the tests assert, along
    with the Wien displacement law lam_max T = b with
    b = 2.897771955e-3 m K (CODATA 2018 exact-constant derivation) and
    the Stefan-Boltzmann integral.

    Parameters
    ----------
    lam : array-like
        Wavelengths (m), > 0.
    T : float
        Temperature (K), > 0.
    h, c, kB : float
        Planck constant (J s), speed of light (m/s), Boltzmann
        constant (J/K) -- 2019 SI exact values by default.

    Returns
    -------
    result : RichResult
        Keys: estimate (spectral radiance per wavelength),
        peak_wavelength (Wien), total_power (Stefan-Boltzmann,
        W m-2), lam, T, method.

    References
    ----------
    Planck, M. (1900), "Zur Theorie des Gesetzes der
    Energieverteilung im Normalspectrum", Verhandlungen der Deutschen
    Physikalischen Gesellschaft 2, 237-245 (the law). Constants: 2019
    SI redefinition (BIPM), CODATA 2018. Wavelength-form statement as
    in any standard radiometry text; consistency with the in-tree
    frequency form (morie.fn.plank) is asserted exactly.
    """
    lam_a = np.asarray(lam, dtype=float)
    T = float(T)
    if T <= 0:
        raise ValueError("plncF: temperature must be > 0")
    vals = []
    for l in lam_a.ravel():
        l = float(l)
        if l <= 0:
            raise ValueError("plncF: wavelengths must be > 0")
        x = h * c / (l * kB * T)
        x = min(x, 700.0)
        B = (2.0 * h * c * c / l ** 5) / (math.exp(x) - 1.0 + 1e-300)
        vals.append(B)
    # Wien displacement: lam_max = b / T, b = hc / (kB x*) with
    # x* = 4.965114231744276 the root of (x-5)e^x + 5 = 0
    xstar = 4.965114231744276
    b_wien = h * c / (kB * xstar)
    lam_peak = b_wien / T
    sigma = 2.0 * math.pi ** 5 * kB ** 4 / (15.0 * h ** 3 * c ** 2)
    return RichResult(payload={
        "estimate": vals,
        "peak_wavelength": lam_peak,
        "wien_constant": b_wien,
        "total_power": sigma * T ** 4,
        "T": T,
        "method": "Planck spectral radiance, wavelength form B(lam,T) = 2hc^2/lam^5 / (exp(hc/lam kB T)-1)",
    })


plncF = planck_function


def cheatsheet():
    return "plncF(lam, T) -> Planck B(lam,T); Wien peak b/T; consistent with plank (frequency form) via c/lam^2 Jacobian"

# public names resolved by fn/_lazy_map.json
planckfunction = planck_function
