# morie.fn -- function file (rootcoder007/morie)
r"""ReLU6 activation function.

Clamps ReLU output to [0, 6].

References
----------
Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).
MobileNetV2: Inverted residuals and linear bottlenecks.
In CVPR (pp. 4510-4520).
"""

__all__ = ["relu6"]

from . import _array_core as np


def relu6(x, derivative=False):
    """
    ReLU6 activation.

    Parameters
    ----------
    x : ndarray
        Input.
    derivative : bool, optional
        Return gradient. Default False.

    Returns
    -------
    ndarray
        Output or gradient.
    """
    x_in = x
    x = np.asarray(x, dtype=float)

    if derivative:
        return np.scalar_out(x_in, np.where((x >= 0) & (x <= 6), 1.0, 0.0))
    else:
        return np.scalar_out(x_in, np.clip(x, 0, 6))


def cheatsheet() -> str:
    return "relu6: relu6(x, derivative) -> ReLU6 activation."
