# morie.fn -- slice s03 (rootcoder007/morie)
"""Sparse GP with M inducing points, FITC and DTC side by side.

Sources consulted: Snelson, E. and Ghahramani, Z. (2006), *NIPS* 18,
1257-1264 (FITC); and Quinonero-Candela, J. and Rasmussen, C. E. (2005).
A unifying view of sparse approximate Gaussian process regression.
*JMLR* 6, 1939-1959, whose section 6 places DTC and FITC in one family:
both replace K_nn by the Nystrom term Q_nn = K_nm K_mm^(-1) K_mn, and
they differ only in the diagonal,

    DTC : Q_nn
    FITC: Q_nn + diag( K_nn - Q_nn )

so FITC alone reproduces the exact marginal variances.  Neither source
was retrievable here as a full text; the two covariances are quoted in
their standard published form.

Inducing points default to an even subsample of the training inputs
taken by index, not by a random draw -- the choice is deterministic and
both arms make it identically.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as k

from ._richresult import RichResult

from .gpsfn import gp_sparse_inducing

__all__ = ["sparse_gp"]


def sparse_gp(X, y, M=3, X_test=None, gamma=1.0, sigma2=1e-2):
    """FITC and DTC predictions from M evenly-spaced inducing points.

    Returns
    -------
    RichResult with payload:
        estimate : the FITC prediction at the first test point
        pred_fitc, pred_dtc, var_fitc, var_dtc
        inducing : the indices used
    """
    Xm = k.mat(X)
    n = len(Xm)
    m = int(M)
    if m > n:
        m = n
    idx = [int(round(t * (n - 1) / (m - 1))) if m > 1 else 0 for t in range(m)]
    Z = [Xm[i] for i in idx]
    f = gp_sparse_inducing(X, y, X_test, Z, gamma, sigma2, 1e-8, "fitc")
    d = gp_sparse_inducing(X, y, X_test, Z, gamma, sigma2, 1e-8, "dtc")
    return RichResult(
        title="Sparse GP",
        summary_lines=[("inducing", m)],
        payload={
            "estimate": f["pred"][0] if f["pred"] else float("nan"),
            "pred_fitc": f["pred"],
            "pred_dtc": d["pred"],
            "var_fitc": f["var"],
            "var_dtc": d["var"],
            "inducing": idx,
            "method": "FITC and DTC sparse GP (Snelson and Ghahramani 2006; Quinonero-Candela and Rasmussen 2005)",
        },
    )


def cheatsheet():
    return "sgpr: FITC / DTC sparse GP"


sparsegp = sparse_gp
