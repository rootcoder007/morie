# morie.fn -- function file (rootcoder007/morie)
"""Andrews sine IRLS weight function."""

import math

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['andrewswt', 'andrews_sine', 'andrewssine']


def andrewswt(r, A=1.339):
    """Andrews sine IRLS weight function.

    The IRLS weight is psi(r)/r, which at r = 0 is a removable singularity with limit one; evaluating it naively gives a division by zero at exactly the observation that fits best, so the limit is taken explicitly.


    Formula: w(r) = sin(r/A)/(r/A) for |r| <= A pi, and 0 otherwise; w(0) = 1

    Parameters
    ----------
    r : array-like
        Scaled residuals.
    A : float
        Tuning constant; 1.339 gives 95% Gaussian efficiency.

    Returns
    -------
    RichResult
        ``weight``, ``rejected``, ``A``, ``n``.

    References
    ----------
    Andrews (1974), A robust method for multiple linear regression,
    Technometrics 16:523-531.  Not held locally; w(z) = sin(z/A)/(z/A)
    for |z| <= A pi with A = 1.339 is as documented by statsmodels'
    AndrewWave norm, the reference implementation.
    """
    r = C.vec(r)
    A = float(A)
    if A <= 0:
        raise ValueError("A must be positive")
    lim = A * math.pi
    w = []
    for v in r:
        if abs(v) > lim:
            w.append(0.0)
        elif v == 0.0:
            w.append(1.0)
        else:
            w.append(math.sin(v / A) / (v / A))
    return RichResult(payload={
        "weight": w, "rejected": sum(1 for v in r if abs(v) > lim),
        "A": A, "n": len(r), "method": "Andrews sine IRLS weight"})


andrews_sine = andrewswt
andrewssine = andrewswt


def cheatsheet():
    return "andrew: Andrews sine IRLS weight function."
