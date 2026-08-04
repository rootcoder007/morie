# morie.fn -- function file (rootcoder007/morie)
"""Identification of beta and G in a semiparametric single-index model.

Horowitz (2009), *Semiparametric and Nonparametric Methods in
Econometrics*, Springer Series in Statistics, Section 2.3.1,
Theorem 2.1 (pages 12-14).

Model (2.1):  E(Y | X = x) = G(x' beta).

Theorem 2.1 -- beta and G are identified if

  (a) G is differentiable and not constant on the support of X'beta;
  (b) the components of X are continuously distributed with a joint
      density;
  (c) the support of X is not contained in any proper linear subspace
      of R^d;
  (d) beta_1 = 1.

(a) is a property of the unknown G, so it can only be checked against
the data through the observed variation of Y along the index -- which
is what the `y` argument is for. (b), (c) and (d) are checkable
directly from X and beta, and that is what this function does.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["simident", "horowitz_sim_identification"]


def simident(x, beta, y=None, mindistinct=10):
    """Check the Theorem 2.1 conditions for single-index identification.

    Parameters
    ----------
    x : array-like, (n, d)
        Covariate matrix.  Must contain NO constant column: a constant
        component is the location normalisation the theorem removes
        (Horowitz 2009, page 13).
    beta : array-like, (d,)
        Index coefficients.  Scale normalisation is beta[0] = 1.
    y : array-like, (n,), optional
        Outcome.  When supplied, the spread of Y across index deciles
        is reported as the observable evidence bearing on condition
        (a) -- G nonconstant.  Not supplied, condition (a) is reported
        as unchecked.
    mindistinct : int, default 10
        A column of X is treated as continuously distributed when it
        takes at least this many distinct values.  Deterministic, and
        a fixed default rather than a data-dependent threshold.

    Returns
    -------
    RichResult
        payload keys: identified, conda, condb, condc, condd, rank,
        dim, minsv, ncontin, gspread, nconstcol, n, method.
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    b = np.asarray(beta, dtype=float).ravel()
    if X.shape[1] != b.size and X.shape[0] == b.size:
        X = X.T
    n, d = X.shape

    # (d) scale normalisation beta_1 = 1
    condd = bool(abs(float(b[0]) - 1.0) <= 1e-12)

    # (c) support not in a proper linear subspace of R^d: the design
    # must have full column rank d.  Reported with the smallest
    # singular value, which is how close it is to failing.
    sv = np.linalg.svd(X, compute_uv=False)
    rank = int(np.sum(sv > sv[0] * 1e-12)) if sv.size and sv[0] > 0 else 0
    minsv = float(sv[-1]) if sv.size else 0.0
    condc = bool(rank == d)

    # a constant column is the location normalisation the theorem bars
    nconstcol = int(sum(1 for j in range(d) if float(np.std(X[:, j])) <= 0.0))

    # (b) continuously distributed components
    ncontin = int(sum(1 for j in range(d)
                      if np.unique(X[:, j]).size >= mindistinct))
    condb = bool(ncontin == d)

    # (a) G nonconstant -- observable proxy: variation of the mean of Y
    # across deciles of the fitted index.  Zero spread is evidence that
    # G is flat; positive spread is not proof that it is not.
    z = X @ b
    if y is None:
        conda = None
        gspread = float("nan")
    else:
        yv = np.asarray(y, dtype=float).ravel()
        cuts = np.quantile(z, np.linspace(0.0, 1.0, 11))
        means = []
        for k in range(10):
            lo, hi = float(cuts[k]), float(cuts[k + 1])
            sel = (z >= lo) & (z <= hi) if k == 9 else (z >= lo) & (z < hi)
            if bool(np.any(sel)):
                means.append(float(np.mean(yv[sel])))
        gspread = float(max(means) - min(means)) if means else 0.0
        conda = bool(gspread > 0.0)

    identified = bool(condb and condc and condd and nconstcol == 0
                      and (conda is not False))
    return RichResult(
        title="Single-index identification (Theorem 2.1)",
        payload={
            "identified": identified,
            "conda": conda,
            "condb": condb,
            "condc": condc,
            "condd": condd,
            "rank": rank,
            "dim": d,
            "minsv": minsv,
            "ncontin": ncontin,
            "gspread": gspread,
            "nconstcol": nconstcol,
            "n": n,
            "method": "Horowitz (2009) Theorem 2.1 identification conditions",
        },
    )


# canonical long-form spelling kept working as an alias
horowitz_sim_identification = simident


def cheatsheet():
    return "hrzsic: single-index identification conditions (Theorem 2.1)"


# CANONICAL TEST
if __name__ == "__main__":  # pragma: no cover
    X = np.column_stack([np.linspace(-2, 2, 200),
                         np.linspace(3, -1, 200) ** 2])
    r = simident(X, [1.0, 0.5])
    assert r["condc"] and r["condd"] and r["condb"], r
    # a rank-deficient design fails condition (c)
    X2 = np.column_stack([X[:, 0], 2.0 * X[:, 0]])
    assert not simident(X2, [1.0, 0.5])["condc"]
    # a wrong scale normalisation fails condition (d)
    assert not simident(X, [2.0, 0.5])["condd"]
    print("ok")
