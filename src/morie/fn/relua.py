# morie.fn -- function file (rootcoder007/morie)
"""Rectifier linear unit activation and its gradient."""

from . import _tail1core as C

from ._richresult import RichResult

__all__ = ['reluact', 'relu_activation', 'reluactivation']


def reluact(z, slope=0.0):
    """Rectifier linear unit activation and its gradient.

    Formula: g(z) = max(0, z);  g'(z) = 1 if z > 0, else 0

    Parameters
    ----------
    z : array-like
        Pre-activation values.
    slope : float
        Slope applied below the threshold; 0 gives the plain ReLU, a small positive value gives the leaky ReLU of Sect. 10.3.3.

    Returns
    -------
    RichResult
        ``activation``, ``gradient``, ``n``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), Multivariate Statistical Machine Learning Methods for Genomic Prediction, Springer, doi:10.1007/978-3-030-89010-0.  Chapter 10, Sect. 10.3.2 p. 388: g(z) = max(0, z), flat below the threshold and linear above it; Sect. 10.3.3 p. 389 gives the leaky variant with slope alpha below zero (the figure uses alpha = 0.1).  Read from the chapter PDF, not recalled.
    """
    z = C.vec(z)
    slope = float(slope)
    act = [v if v > 0.0 else slope * v for v in z]
    grd = [1.0 if v > 0.0 else slope for v in z]
    return RichResult(payload={
        "activation": act, "gradient": grd, "n": len(z),
        "method": "ReLU activation, MVSML Sect. 10.3.2"})


relu_activation = reluact
reluactivation = reluact


def cheatsheet():
    return 'relua: Rectifier linear unit activation and its gradient.'
