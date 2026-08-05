# morie.fn -- function file (rootcoder007/morie)
"""Forecast error variance decomposition from a VAR(1) model."""

from . import _array_core as np

from ._containers import DescriptiveResult


def fevd(
    var_coefficients: np.ndarray,
    sigma_u: np.ndarray,
    periods: int = 20,
) -> DescriptiveResult:
    """
    Forecast error variance decomposition from a VAR(1).

    Computes the fraction of forecast error variance of each variable
    attributable to shocks in each variable, using the Cholesky
    decomposition of the residual covariance.

    :param var_coefficients: VAR(1) coefficient matrix A (k x k).
    :param sigma_u: Residual covariance matrix (k x k).
    :param periods: Forecast horizon. Default 20.
    :return: DescriptiveResult with decomposition array (periods, k, k).
    :raises ValueError: If matrices are not conformable.

    References
    ----------
    Lutkepohl H. (2005). New Introduction to Multiple Time Series
    Analysis. Springer.
    """
    A = np.asarray(var_coefficients, dtype=float)
    S = np.asarray(sigma_u, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("var_coefficients must be square.")
    k = A.shape[0]
    if S.shape != (k, k):
        raise ValueError("sigma_u must be k by k.")
    periods = int(periods)
    if periods < 0:
        raise ValueError("periods must be non-negative.")
    # Plain nested lists: _array_core does not support assigning a 2-D
    # block into a 3-D array, which silently made this module unusable
    # (ValueError: row assignment length mismatch on the first call).
    Am = [[float(A[i][j]) for j in range(k)] for i in range(k)]
    P = [[float(v) for v in row] for row in np.linalg.cholesky(S)]
    Theta = [[[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]]
    A_pow = [[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)]
    for _h in range(1, periods + 1):
        A_pow = [[sum(A_pow[i][t] * Am[t][j] for t in range(k)) for j in range(k)]
                 for i in range(k)]
        Theta.append([row[:] for row in A_pow])
    mse = []
    decomp = []
    for h in range(periods + 1):
        contrib = [[0.0] * k for _ in range(k)]
        for s_ in range(h + 1):
            TP = [[sum(Theta[s_][i][t] * P[t][j] for t in range(k)) for j in range(k)]
                  for i in range(k)]
            for i in range(k):
                for j in range(k):
                    contrib[i][j] += TP[i][j] ** 2
        total_var = [sum(contrib[i]) for i in range(k)]
        dec = [[0.0] * k for _ in range(k)]
        for j in range(k):
            for i in range(k):
                dec[i][j] = contrib[i][j] / total_var[i] if total_var[i] > 0 else 0.0
        decomp.append(dec)
        mse.append(contrib)
    return DescriptiveResult(
        name="fevd",
        value=float(decomp[-1][0][0]),
        extra={
            "decomposition": decomp,
            "mse_contributions": mse,
            "periods": periods,
            "k": k,
        },
    )


fevdc = fevd


def cheatsheet() -> str:
    return "fevd({}) -> Forecast error variance decomposition."
