# morie.fn -- function file (rootcoder007/morie)
"""Tukey biweight rho, psi and weight functions."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["tukey_biweight"]


def tukey_biweight(r, c=4.685):
    """The biweight loss and its derivative, evaluated pointwise.

    This is the objective itself rather than an estimator: ``rho`` is
    the loss an M-estimator minimises, ``psi`` its derivative, ``w``
    the IRLS weight ``psi(r) / r``.  ``rho`` is flat past ``c``, which
    is the whole design -- a residual beyond the tuning constant costs
    a constant, so pushing it further out costs nothing and the fit
    stops chasing it.

    Formula: ``rho(r) = (c^2 / 6)(1 - [1 - (r / c)^2]^3)`` for
    ``|r| <= c`` and ``c^2 / 6`` otherwise;
    ``psi(r) = r [1 - (r / c)^2]^2`` inside and 0 outside.

    Parameters
    ----------
    r : array-like
        Scaled residuals.
    c : float, default 4.685
        Tuning constant.

    Returns
    -------
    RichResult
        ``estimate`` (total loss ``sum rho``), ``rho``, ``psi``, ``w``,
        ``n``.

    References
    ----------
    Beaton, A. E. & Tukey, J. W. (1974).  The fitting of power series,
    meaning polynomials, illustrated on band-spectroscopic data.
    Technometrics 16:147-185.
    """
    v = C.vec(r)
    c = float(c)
    cap = c * c / 6.0
    rho, psi, w = [], [], []
    for t in v:
        u = t / c
        if abs(u) <= 1.0:
            rho.append(cap * (1.0 - (1.0 - u * u) ** 3))
            psi.append(t * (1.0 - u * u) ** 2)
            w.append((1.0 - u * u) ** 2)
        else:
            rho.append(cap)
            psi.append(0.0)
            w.append(0.0)
    return RichResult(payload={
        "estimate": sum(rho), "rho": rho, "psi": psi, "w": w, "n": len(v),
        "method": "Tukey biweight rho, psi and weight"})


def cheatsheet():
    return "tukrho: Tukey biweight rho, psi and weight functions."
