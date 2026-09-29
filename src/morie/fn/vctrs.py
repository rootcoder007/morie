# morie.fn -- function file (rootcoder007/morie)
"""Generate neural network weight initialization matrices."""

from __future__ import annotations

from . import _winit as wi
from ._containers import DescriptiveResult


def weight_init(
    fan_in: int,
    fan_out: int,
    *,
    method: str = "xavier_uniform",
    gain: float = 1.0,
    seed: int = 42,
) -> DescriptiveResult:
    r"""Generate neural network weight initialization matrices.

    Variance-preserving initialisations of a ``fan_in x fan_out`` layer:
    Xavier/Glorot (Glorot and Bengio 2010) ``U(+-gain sqrt(6/(fi + fo)))``
    or ``N(0, gain^2 2/(fi + fo))``; He/Kaiming (He et al. 2015)
    ``U(+-gain sqrt(6/fi))`` or ``N(0, gain^2 2/fi)``; LeCun ``N(0, gain^2
    / fi)``; and orthogonal (Saxe, McClelland and Ganguli 2014): a Gaussian
    matrix orthonormalised by modified Gram-Schmidt (columns when ``fi >=
    fo``, rows otherwise), times ``gain``. Draws are Philox uniforms or
    normals (``seed``), filled row by row, so the R twin reproduces them.

    Parameters
    ----------
    fan_in, fan_out : int
        Layer sizes.
    method : str
        ``xavier_uniform``, ``xavier_normal``, ``he_uniform``, ``he_normal``,
        ``lecun_normal`` or ``orthogonal``.
    gain : float
        Scaling factor.
    seed : int
        Philox seed.

    Returns
    -------
    DescriptiveResult
        ``value`` is the weight matrix (list of rows); ``extra`` has the
        empirical ``variance`` and the target ``expected_variance`` (``gain^2``
        times the method's variance).

    References
    ----------
    Glorot, X. and Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural
    networks. *AISTATS*, 249-256.

    He, K., Zhang, X., Ren, S. and Sun, J. (2015). Delving deep into rectifiers. *ICCV*, 1026-1034.

    Saxe, A. M., McClelland, J. L. and Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of
    learning in deep linear neural networks. *ICLR*.

    Examples
    --------
    >>> r = weight_init(3, 2, method="xavier_uniform", seed=1)
    >>> [[round(v, 12) for v in row] for row in r.value]
    [[0.855015009047, 0.864710888758], [0.187816142658, 0.462865164843], [0.376829326712, 0.819638183648]]
    """
    if fan_in < 1 or fan_out < 1:
        raise ValueError("fan_in and fan_out must be positive")
    if method not in wi.VARIANCE:
        raise ValueError(f"Unknown method: {method}")
    W = wi.draw(fan_in, fan_out, method, float(gain), seed, fan_in, fan_out)
    _m, var = wi.moments(W)
    return DescriptiveResult(
        name="weight_init",
        value=W,
        extra={
            "method": method,
            "fan_in": fan_in,
            "fan_out": fan_out,
            "variance": var,
            "expected_variance": float(gain) ** 2 * wi.VARIANCE[method](fan_in, fan_out),
            "gain": gain,
        },
    )


vctrs = weight_init


def cheatsheet() -> str:
    return "weight_init(fan_in, fan_out, method, gain, seed) -> Xavier / He / LeCun / orthogonal weights (Philox)"


# compact alias per ledger/NAMING.md
weightinit = weight_init
