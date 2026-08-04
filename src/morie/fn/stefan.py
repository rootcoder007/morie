# morie.fn -- slice s03 (rootcoder007/morie)
"""Stefan-Boltzmann radiant exitance.

Source consulted: Stefan, J. (1879).  Ueber die Beziehung zwischen der
Waermestrahlung und der Temperatur.  *Sitzungsberichte der Kaiserlichen
Akademie der Wissenschaften* 79, 391-428; Boltzmann, L. (1884).
Ableitung des Stefan'schen Gesetzes.  *Annalen der Physik* 258, 291-294.
The law is

    j* = epsilon sigma T^4

with sigma the Stefan-Boltzmann constant.  Since the 2019 redefinition
of the SI base units sigma is exact,

    sigma = 2 pi^5 k_B^4 / (15 h^3 c^2)
          = 5.670374419e-8 W m^-2 K^-4

(CODATA 2018 / SI 2019, exact by definition of k_B, h and c).  The
constant is computed here from k_B, h and c rather than pasted, so that
the value is traceable and identical in both arms.
"""

from __future__ import annotations

import math

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

__all__ = ["stefan_boltzmann"]

# SI 2019 exact defining constants.
_K_B = 1.380649e-23  # J K^-1
_H = 6.62607015e-34  # J s
_C = 299792458.0  # m s^-1


def _sigma():
    return 2.0 * math.pi ** 5 * _K_B ** 4 / (15.0 * _H ** 3 * _C ** 2)


def stefan_boltzmann(T, emissivity=1.0):
    """Radiant exitance of a grey body at absolute temperature T.

    Parameters
    ----------
    T : float or array-like
        Absolute temperature(s) in kelvin.
    emissivity : float
        Emissivity in [0, 1]; 1 is a black body.

    Returns
    -------
    RichResult with payload:
        estimate  : j* for the first (or only) temperature, W m^-2
        exitance  : list of j* for every temperature supplied
        sigma     : the Stefan-Boltzmann constant used
        total     : sum of the exitances
    """
    t = k.vec(T)
    eps = float(emissivity)
    sig = _sigma()
    out = [eps * sig * x ** 4 for x in t]
    tot = 0.0
    for v in out:
        tot += v
    return RichResult(
        title="Stefan-Boltzmann law",
        summary_lines=[("sigma (W m^-2 K^-4)", sig)],
        payload={
            "estimate": out[0] if out else float("nan"),
            "exitance": out,
            "sigma": sig,
            "emissivity": eps,
            "total": tot,
            "n": len(t),
            "method": "Stefan-Boltzmann radiant exitance j* = eps sigma T^4",
        },
    )


def cheatsheet():
    return "stefan: Stefan-Boltzmann radiation"
