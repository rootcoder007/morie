# morie.fn -- function file (rootcoder007/morie)
"""Nonparametric binary regression with a GP prior.

Implements sec. 2.5 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_np_binary_reg"]


def ghosal_np_binary_reg(x, y, length=0.7, var=2.0):
    """P(Y=1|x) = Phi(f(x)), f ~ GP (GvdV 2017 sec. 2.5). MAP fit by
    Newton iterations on the probit-GP log-posterior (the Laplace
    route the chapter's computation sections use)."""
    import math
    from ._stats_core import norm as _norm
    xs = _bnp._flat(x)
    ys = _bnp._flat(y)
    n = len(xs)
    k = _bnp.rbf_kernel(length, var)
    K = [[k(xs[i], xs[j]) + (1e-8 if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    Ki = np.linalg.pinv(np.marr(K)).tolist()
    f = [0.0] * n
    for _ in range(50):
        grad = []
        Wd = []
        for i in range(n):
            phi = math.exp(-0.5 * f[i] * f[i]) / math.sqrt(2 * math.pi)
            Phi = float(_norm.cdf(f[i]))
            Phi = min(max(Phi, 1e-10), 1 - 1e-10)
            s = phi / Phi if ys[i] > 0.5 else -phi / (1 - Phi)
            grad.append(s)
            Wd.append(s * (s + f[i]) if True else 0.0)
        # Newton step on log posterior: (K^-1 + W) df = grad - K^-1 f
        A = [[Ki[i][j] + (Wd[i] if i == j else 0.0)
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
    p = [float(_norm.cdf(v)) for v in f]
    res = RichResult(payload={"estimate": sum(p) / n, "prob": p,
                              "f": f,
                              "method": "probit-GP binary regression, Laplace MAP (GvdV 2017 sec. 2.5)"})
    return with_describe_pointer(res, "gh_c2_9")


def cheatsheet():
    return "gh_c2_9: Nonparametric binary regression with a GP prior"
