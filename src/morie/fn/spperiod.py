# SPDX-License-Identifier: AGPL-3.0-or-later
"""Periodogram of a process observed on a rectangular lattice."""

import numpy as np

from ._richresult import RichResult
from ._schab_spectral import (fourier_frequencies, periodogram,
                              periodogram_from_covariance, sample_covariance_2d)

__all__ = ["schabenberger_periodogram"]


def schabenberger_periodogram(z_lattice, coords=None, omit_zero_frequency=True,
                              check_identity=True):
    """The periodogram on an r x c lattice, Sec. 4.7.1.

    eq (4.57)::

        I(w1,w2) = 1/{(2 pi)^2 r c}
                   | sum_u sum_v Z(u,v) exp{-i(w1 u + w2 v)} |^2

    evaluated at the Fourier frequencies, which are the multiples of
    ``2 pi / r`` and ``2 pi / c`` running from ``-floor((n-1)/2)`` to
    ``floor(n/2)``.

    The section's central claim is eq (4.59): away from the origin the
    periodogram *is* the Fourier transform of the sample covariance
    function. With ``check_identity`` the right-hand side is computed
    independently and the largest discrepancy is returned as
    ``identity_max_abs_diff``; it should be at machine precision. That check
    is what pins the ``(2 pi)^2`` normalisation, which the stub this replaces
    had dropped in favour of ``1/n``.

    ``coords`` is accepted and ignored: the estimator is defined on the
    row-column lattice, not on arbitrary coordinates.

    Parameters
    ----------
    z_lattice : array-like, shape (r, c)
    coords : ignored
    omit_zero_frequency : bool, default True
        Remove the mean first. At a non-zero Fourier frequency this changes
        nothing (p. 191: ``sum_u cos(w_j u) = 0``); at the origin it removes
        the squared mean, which is where eq (4.59) does not apply.
    check_identity : bool, default True

    Returns
    -------
    RichResult
        Keys: ``periodogram``, ``omega1``, ``omega2``, ``j``, ``k``,
        ``covariance``, ``mean_invariant``, ``r``, ``c``, and when checked
        ``identity_max_abs_diff`` and ``identity_holds``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005), Sec. 4.7.1, eqs (4.56)-(4.59),
    pp. 190-192.
    """
    p = periodogram(z_lattice, omit_zero_frequency=omit_zero_frequency)
    cov, lags_j, lags_k = sample_covariance_2d(z_lattice)
    payload = {
        "periodogram": p["periodogram"],
        "omega1": p["omega1"],
        "omega2": p["omega2"],
        "j": p["j"],
        "k": p["k"],
        "covariance": cov,
        "lags_j": lags_j,
        "lags_k": lags_k,
        "mean_invariant": p["mean_invariant"],
        "nonzero_mask": p["nonzero_mask"],
        "r": p["r"],
        "c": p["c"],
    }
    lines = [("lattice", "%d x %d" % (p["r"], p["c"])),
             ("frequencies", "%d x %d" % (p["omega1"].size, p["omega2"].size)),
             ("mean-invariant off the origin", p["mean_invariant"])]
    if check_identity:
        q = periodogram_from_covariance(z_lattice)
        m = p["nonzero_mask"]
        d = float(np.abs(p["periodogram"][m] - q["periodogram"][m]).max())
        payload["identity_max_abs_diff"] = d
        payload["identity_holds"] = bool(d < 1e-8)
        payload["periodogram_from_covariance"] = q["periodogram"]
        lines.append(("eq (4.59) max|difference|", d))
        if not payload["identity_holds"]:
            payload["warning"] = (
                "the periodogram does not match the Fourier transform of the "
                "sample covariance function; eq (4.59) should hold to machine "
                "precision away from the origin")
    return RichResult(title="Periodogram on a rectangular lattice",
                      summary_lines=lines, payload=payload)


def cheatsheet():
    return ("spperiod: periodogram on an r x c lattice at the Fourier "
            "frequencies (Sec. 4.7.1, eqs (4.57) and (4.59))")
