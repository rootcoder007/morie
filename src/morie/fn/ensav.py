"""Ensemble average."""

from __future__ import annotations

import math

from ._containers import DescriptiveResult


def _is2d(a):
    rows = a.tolist() if hasattr(a, "tolist") else list(a)
    return bool(rows) and hasattr(rows[0], "__len__")


def _vec(x):
    return [float(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def ensemble_average(segments):
    r"""Ensemble (synchronized) average ``ybar(n) = (1/M) sum_k y_k(n)`` of ``M`` time-locked sweeps of length ``N`` (rows); averaging reduces uncorrelated noise variance by ``1/M``. Returns the averaged signal as a list.

    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis*, 2nd ed., sec. 3.1.
    Wiley-IEEE Press.

    Examples
    --------
    >>> ensemble_average([[1.0, 2.0, 3.0], [3.0, 2.0, 5.0]]).value
    [2.0, 2.0, 4.0]
    """
    rows = (
        [_vec(r) for r in (segments.tolist() if hasattr(segments, "tolist") else segments)]
        if _is2d(segments)
        else [_vec(segments)]
    )
    M, N = len(rows), len(rows[0])
    avg = [math.fsum(r[j] for r in rows) / M for j in range(N)]
    return DescriptiveResult(name="ensemble_average", value=avg, extra={"M": M, "N": N})


ensav = ensemble_average


def cheatsheet() -> str:
    return "ensemble_average(segments) -> (1/M) sum_k y_k(n)."
