# morie.fn -- function file (rootcoder007/morie)
"""GWR hat matrix diagonal (leverage)."""

from ._qpcore import ssum
from .gwrcoef import _setup, _wls


def gwrhat(y, X, coords, bw=0.5, kernel="bisquare", adaptive=False):
    r"""Leverages ``S_ii = x_i^T (X^T W_i X)^{-1} X^T W_i e_i``, the diagonal of the GWR hat matrix ``S`` with ``y_hat = S y``; ``tr S`` is the effective number of parameters (Fotheringham, Brunsdon and Charlton 2002, sec. 4.4).

    References
    ----------
    Fotheringham, A. S., Brunsdon, C. and Charlton, M. (2002).
    *Geographically Weighted Regression*. Wiley.

    Examples
    --------
    >>> P = [(float(i % 4), float(i // 4)) for i in range(16)]
    >>> X = [[(0.3 * i) % 1.7] for i in range(16)]
    >>> y = [1.0 + 2.0 * X[i][0] + 0.1 * P[i][0] + 0.05 * (i % 3) for i in range(16)]
    >>> round(sum(gwrhat(y, X, P, 3.0, kernel="gaussian")), 10)
    2.517563543
    """
    yv, Xm, _, _, Wc = _setup(y, X, coords, bw, kernel, adaptive)
    out = []
    for i, w in enumerate(Wc):
        _, C = _wls(Xm, w, yv)
        out.append(ssum(Xm[i][a] * C[a][i] for a in range(len(Xm[0]))))
    return out


gwrhat_fn = gwrhat


def cheatsheet() -> str:
    return "gwrhat(y, X, coords, bw) -> diagonal of the GWR hat matrix (leverages)."
