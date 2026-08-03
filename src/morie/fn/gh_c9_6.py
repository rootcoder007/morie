# morie.fn -- function file (rootcoder007/morie)
"""Location-scale (Wishart) DPM.

Implements sec. 9.4.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, CUP.
"""

import math

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_wishart_dpm"]


def ghosal_wishart_dpm(x_query=(0.0, 1.0), n_atoms=150, alpha=1.0,
                       nu=4.0, seed=42):
    """Location-scale DPM with a Wishart-type scale prior attains
    the same rate as fixed-scale mixtures (sec. 9.4.5). Draws a
    stick-breaking mixture with inverse-gamma scales (1-d Wishart
    margin) and evaluates the density -- positive, normalized.
    Keys: estimate."""
    rng = np.random.default_rng(seed)
    V = [float(rng.beta(1.0, alpha)) for _ in range(n_atoms)]
    W = _bnp.stick_breaking(V)
    mus = [float(rng.normal(0, 1)) for _ in range(n_atoms)]
    sig2 = [nu / max(float(rng.gamma(nu / 2.0, 2.0)), 1e-9)
            for _ in range(n_atoms)]
    def dens(x):
        tot = 0.0
        for w, m, s2 in zip(W, mus, sig2):
            tot += w * math.exp(-0.5 * (x - m) ** 2 / s2) \
                / math.sqrt(2.0 * math.pi * s2)
        return tot
    vals = [dens(q) for q in x_query]
    # normalization by quadrature
    Z = sum(dens(-8.0 + 16.0 * (i + 0.5) / 400) for i in range(400)) \
        * 16.0 / 400
    res = RichResult(payload={"estimate": vals[0],
                              "density": vals,
                              "total_mass": Z,
                              "method": "location-scale DPM (GvdV 2017 sec. 9.4.5)"})
    return with_describe_pointer(res, "gh_c9_6")


def cheatsheet():
    return "gh_c9_6: Location-scale (Wishart) DPM"
