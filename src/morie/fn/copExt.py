# morie.fn -- function file (rootcoder007/morie)
"""Extreme-value copula from a Pickands dependence function."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["extremal_copula"]


def _pickands_builtin(name, theta):
    if name == "gumbel":
        if theta < 1:
            raise ValueError("gumbel/logistic theta must be >= 1.")
        return lambda t: (t**theta + (1 - t) ** theta) ** (1.0 / theta)
    if name == "galambos":
        if theta <= 0:
            raise ValueError("galambos delta must be positive.")
        return lambda t: 1.0 - (t ** (-theta) + (1 - t) ** (-theta)) ** (-1.0 / theta)
    if name == "independence":
        return lambda t: np.ones_like(np.asarray(t, dtype=float))
    raise ValueError("A must be callable or one of 'gumbel', 'galambos', 'independence'.")


def extremal_copula(u, v, A="gumbel", theta=2.0):
    r"""Bivariate extreme-value copula.

    .. math:: C(u, v) = \exp\left\{ \ln(uv)\,
              A\!\left(\frac{\ln v}{\ln(uv)}\right) \right\},

    with A the Pickands dependence function -- convex on [0, 1] with
    :math:`\max(t, 1-t) \le A(t) \le 1`. ``A`` may be a callable or
    one of the built-ins from Czado's Table 3.1: ``gumbel`` (the
    logistic model, :math:`A(t) = [t^\theta + (1-t)^\theta]^{1/\theta}`,
    verified against that table), ``galambos``, ``independence``.

    Every extreme-value copula is max-stable:
    :math:`C(u^k, v^k) = C(u, v)^k` for all k > 0, which the tests
    assert directly.

    Parameters
    ----------
    u, v : array-like in (0, 1)
        Uniform margins.
    A : callable or {"gumbel", "galambos", "independence"}
        Pickands dependence function.
    theta : float, default 2.0
        Parameter for the built-in families.

    Returns
    -------
    RichResult
        keys: ``cdf``, ``pickands_at_half``, ``valid_pickands``
        (bounds check on a grid), ``A``, ``theta``, ``method``.

    References
    ----------
    Czado, C. (2019). *Analyzing Dependent Data with Vine Copulas*.
    Springer. Sec. 3.4, Table 3.1 p. 52 (extreme-value families and
    their Pickands functions), eq. (3.18).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if np.any((u <= 0) | (u >= 1)) or np.any((v <= 0) | (v >= 1)):
        raise ValueError("u and v must lie strictly inside (0, 1).")
    fn = A if callable(A) else _pickands_builtin(A, float(theta))

    grid = np.linspace(0.001, 0.999, 199)
    Avals = np.asarray(fn(grid), dtype=float)
    lower = np.maximum(grid, 1 - grid)
    valid = bool(np.all(Avals <= 1 + 1e-8) and np.all(Avals >= lower - 1e-8))

    luv = np.log(u * v)
    t = np.log(v) / luv
    cdf = np.exp(luv * np.asarray(fn(t), dtype=float))

    return RichResult(
        payload={
            "cdf": cdf,
            "pickands_at_half": float(np.asarray(fn(0.5))),
            "valid_pickands": valid,
            "A": A if isinstance(A, str) else "callable",
            "theta": float(theta),
            "method": "Extreme-value copula from a Pickands dependence function",
        }
    )


def cheatsheet():
    return "copExt: C = exp{ln(uv) A(ln v / ln(uv))}; max-stable by construction"
