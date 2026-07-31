# morie.fn -- function file (rootcoder007/morie)
"""Identification constraints for IRT ideal points."""

import numpy as np

from ._richresult import RichResult

__all__ = ["irt_identification_constraints"]


def irt_identification_constraints(x, polarity_idx=None, pivot_idx=None):
    r"""Normalise a 1-D ideal-point vector to the identified scale.

    The IRT likelihood is invariant to affine maps of the latent
    scale, so estimates only mean anything after fixing location,
    scale, and polarity. This applies the standard normalisation
    (mean 0, sd 1) and then, if ``polarity_idx`` is given, reflects
    the scale so that legislator's position is negative (the
    "liberal-on-the-left" convention); ``pivot_idx`` instead demands
    that legislator positive. Supplying both requires them to be on
    opposite sides after normalisation, else the constraint set is
    infeasible and an error says so.

    Parameters
    ----------
    x : array-like, shape (n,)
        Raw ideal points.
    polarity_idx : int, optional
        Index forced negative.
    pivot_idx : int, optional
        Index forced positive.

    Returns
    -------
    RichResult
        keys: ``x`` (normalised, possibly reflected), ``reflected``,
        ``mean_before``, ``sd_before``, ``n``, ``method``.

    References
    ----------
    Clinton, J., Jackman, S. & Rivers, D. (2004). The statistical
    analysis of roll call data. *APSR*, 98(2), 355-370.
    (identification of the ideal-point model)

    Armstrong, D. A. et al. (2021). *Analyzing Spatial Models of
    Choice and Judgment* (2nd ed.). CRC Press. Ch. 6 (Bayesian scaling
    and identification), p. 181.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 ideal points.")
    sd = x.std()
    if sd <= 0:
        raise ValueError("ideal points are constant; the scale cannot be identified.")
    mu = float(x.mean())
    z = (x - mu) / sd

    reflected = False
    if polarity_idx is not None:
        i = int(polarity_idx)
        if not 0 <= i < n:
            raise ValueError(f"polarity_idx out of range [0, {n - 1}].")
        if z[i] > 0:
            z = -z
            reflected = True
        if z[i] == 0:
            raise ValueError("polarity legislator sits exactly at the mean; pick another.")
    if pivot_idx is not None:
        j = int(pivot_idx)
        if not 0 <= j < n:
            raise ValueError(f"pivot_idx out of range [0, {n - 1}].")
        if polarity_idx is None:
            if z[j] < 0:
                z = -z
                reflected = True
        elif z[j] <= 0:
            raise ValueError(
                "infeasible constraints: polarity and pivot legislators fall on the same side."
            )

    return RichResult(
        payload={
            "x": z,
            "reflected": reflected,
            "mean_before": mu,
            "sd_before": float(sd),
            "n": int(n),
            "method": "IRT identification: mean 0, sd 1, polarity/pivot reflection",
        }
    )


def cheatsheet():
    return "irtid: normalise to mean 0 / sd 1, reflect so polarity_idx < 0 (< pivot_idx)"
