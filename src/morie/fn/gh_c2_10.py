# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric Poisson regression with a GP prior.

Implements sec. 2.6 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_np_poisson_reg"]


def ghosal_np_poisson_reg(x, y, length=0.7, var=1.0):
    """Y|x ~ Poisson(exp f(x)), f ~ GP (GvdV 2017 sec. 2.6). MAP by
    damped Newton on the log-link GP posterior."""
    import math
    xs = _bnp._flat(x)
    ys = _bnp._flat(y)
    n = len(xs)
    k = _bnp.rbf_kernel(length, var)
    K = [[k(xs[i], xs[j]) + (1e-8 if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    Ki = np.linalg.pinv(np.marr(K)).tolist()
    f = [math.log(max(v, 0.5)) for v in ys]
    for _ in range(60):
        lam = [math.exp(min(v, 30.0)) for v in f]
        grad = [ys[i] - lam[i] for i in range(n)]
        A = [[Ki[i][j] + (lam[i] if i == j else 0.0)
              for j in range(n)] for i in range(n)]
        b = [grad[i] - sum(Ki[i][j] * f[j] for j in range(n))
             for i in range(n)]
        try:
            step = np.linalg.solve(np.marr(A), np.marr(b))
        except Exception:
            step = np.linalg.lstsq(np.marr(A), np.marr(b))[0]
        sl = [float(v) for v in step._flat()]
        f = [f[i] + 0.5 * sl[i] for i in range(n)]
        if max(abs(v) for v in sl) < 1e-8:
            break
    lam = [math.exp(v) for v in f]
    res = RichResult(payload={"estimate": sum(lam) / n,
                              "intensity": lam, "f": f,
                              "method": "log-link GP Poisson regression MAP (GvdV 2017 sec. 2.6)"})
    return with_describe_pointer(res, "gh_c2_10")


def cheatsheet():
    return "gh_c2_10: Nonparametric Poisson regression with a GP prior"
