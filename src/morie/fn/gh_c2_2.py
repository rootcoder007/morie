# morie.fn -- function file (rootcoder007/morie)
"""Gaussian-process prior draw.

Implements sec. 2.2.1 of Ghosal & van der Vaart (2017), *Fundamentals of
Nonparametric Bayesian Inference*, Cambridge University Press.
"""

from . import _array_core as np
from . import _bnp_core as _bnp
from ._richresult import RichResult, with_describe_pointer

__all__ = ["ghosal_gp_prior_def"]


def ghosal_gp_prior_def(x, length=0.5, var=1.0, seed=42):
    """f ~ GP(0, k): finite-dimensional draws are multivariate normal
    with covariance k(x_i, x_j) (GvdV 2017 sec. 2.2.1). Draw via the
    Cholesky factor of the squared-exponential kernel matrix."""
    xs = _bnp._flat(x)
    n = len(xs)
    k = _bnp.rbf_kernel(length, var)
    K = [[k(xs[i], xs[j]) + (1e-10 if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    L = np.linalg.cholesky(np.marr(K)).tolist()
    rng = np.random.default_rng(seed)
    z = [float(v) for v in rng.normal(0, 1, n)._flat()]
    f = [sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(n)]
    est = sum(f) / n
    res = RichResult(payload={"estimate": est, "f": f,
                              "method": "GP prior draw via Cholesky (GvdV 2017 sec. 2.2.1)"})
    return with_describe_pointer(res, "gh_c2_2")


def cheatsheet():
    return "gh_c2_2: Gaussian-process prior draw"
