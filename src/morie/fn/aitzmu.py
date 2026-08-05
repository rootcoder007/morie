# morie.fn -- wave 2 slice w2_00 (rootcoder007/morie)
"""Multiplicative replacement of rounded zeros in compositions.

Source: Martin-Fernandez, J. A., Barcelo-Vidal, C. and Pawlowsky-Glahn,
V. (2003), "Dealing with zeros and missing values in compositional data
sets using nonparametric imputation", Mathematical Geology 35(3),
253-278, doi:10.1023/a:1023866030544 (citation verified against
Crossref).

For a composition x with constant sum kappa and zero set Z, the
multiplicative replacement puts

    x'_j = delta_j                                       j in Z
    x'_j = x_j (1 - (sum_{k in Z} delta_k) / kappa)       j not in Z

The rule is built so that (i) the total is preserved exactly,
sum_j x'_j = kappa, and (ii) every ratio between two non-zero parts is
untouched, x'_i / x'_j = x_i / x_j.  Both are exact identities and are
used as the anchors; the second is what makes the replacement
"multiplicative", i.e. a perturbation, rather than an additive fudge
that would distort the log-ratios it is meant to protect.

A row whose parts are all non-zero is returned unchanged.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k  # noqa: F401

from ._richresult import RichResult

__all__ = ["compositional_zero_multreplace"]


def _rows(X):
    """Accept a single composition or a matrix of them; always return rows."""
    try:
        first = X[0]
    except (TypeError, IndexError, KeyError):
        raise ValueError("compositional_zero_multreplace: X is empty")
    if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
        out = [[float(v) for v in r] for r in X]
        if not out:
            raise ValueError("compositional_zero_multreplace: X is empty")
        w = len(out[0])
        for r in out:
            if len(r) != w:
                raise ValueError("compositional_zero_multreplace: X is ragged")
        return out, True
    return [[float(v) for v in X]], False


def compositional_zero_multreplace(X, delta):
    """Replace zero parts by delta, shrinking the observed parts to match.

    Parameters
    ----------
    X : array-like
        One composition, or a matrix whose rows are compositions.  Parts
        must be non-negative; zeros are the ones replaced.
    delta : float or array-like
        The imputed value, either one number for every zero or one per
        part.  Must be strictly positive and small relative to the total.

    Returns
    -------
    X_imp : the imputed rows (same shape as the input)
    n_zero : how many parts were replaced
    """
    rows, was_matrix = _rows(X)
    D = len(rows[0])
    if D < 2:
        raise ValueError("compositional_zero_multreplace: a composition needs at least 2 parts")
    if hasattr(delta, "__len__"):
        dl = [float(v) for v in delta]
        if len(dl) != D:
            raise ValueError("compositional_zero_multreplace: delta has the wrong length")
    else:
        dl = [float(delta)] * D
    for v in dl:
        if not (v > 0.0):
            raise ValueError("compositional_zero_multreplace: delta must be strictly positive")
    out = []
    nz = 0
    for r in rows:
        tot = 0.0
        for v in r:
            if v < 0.0:
                raise ValueError("compositional_zero_multreplace: a part is negative")
            tot += v
        if not (tot > 0.0):
            raise ValueError("compositional_zero_multreplace: a row sums to zero")
        sd = 0.0
        for j in range(D):
            if r[j] == 0.0:
                sd += dl[j]
        if sd >= tot:
            raise ValueError("compositional_zero_multreplace: the imputed mass exceeds the total")
        f = 1.0 - sd / tot
        row = []
        for j in range(D):
            if r[j] == 0.0:
                row.append(dl[j])
                nz += 1
            else:
                row.append(r[j] * f)
        out.append(row)
    return RichResult(
        title="Multiplicative zero replacement",
        summary_lines=[("rows", len(out)), ("replaced", nz)],
        payload={
            "X_imp": out if was_matrix else out[0],
            "estimate": out[0][0],
            "n_zero": nz,
            "n": len(out),
            "D": D,
            "method": "Martin-Fernandez et al. (2003) multiplicative replacement",
        },
    )


def cheatsheet():
    return "aitzmu: Multiplicative replacement of zeros in compositions"


# compact alias per ledger/NAMING.md
compositionalzeromultreplace = compositional_zero_multreplace
