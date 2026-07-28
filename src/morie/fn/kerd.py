# morie.fn -- function file (rootcoder007/morie)
"""Kernel density estimation."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kernel_density"]

KERNELS = ("gaussian", "epanechnikov", "biweight", "uniform", "triangular")


def _k(u, kind):
    a = np.abs(u)
    if kind == "gaussian":
        return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    if kind == "epanechnikov":
        return np.where(a <= 1, 0.75 * (1 - u ** 2), 0.0)
    if kind == "biweight":
        return np.where(a <= 1, (15.0 / 16.0) * (1 - u ** 2) ** 2, 0.0)
    if kind == "uniform":
        return np.where(a <= 1, 0.5, 0.0)
    return np.where(a <= 1, 1 - a, 0.0)


def kernel_density(x, at=None, bandwidth=None, kernel="gaussian",
                   bw_method="silverman"):
    r"""Univariate KDE with a bandwidth chosen visibly.

    .. math:: \hat f(x) = \frac{1}{nh}\sum_{i=1}^{n}
              K\!\left(\frac{x - X_i}{h}\right)

    The bandwidth is the entire estimator. Silverman's rule,
    :math:`h = 0.9\,\min(\hat\sigma, IQR/1.34)\,n^{-1/5}`, minimises
    asymptotic MISE for a GAUSSIAN target, and its use of the IQR is a
    deliberate hedge against the sd being inflated by outliers. On
    genuinely multimodal data it still oversmooths, because it is
    calibrated to a distribution with one mode -- which is precisely
    the case where a density estimate was worth making.

    The kernel choice barely matters: Epanechnikov is MISE-optimal, but
    the efficiency of a Gaussian kernel relative to it is about 95 %,
    so the difference is invisible next to a bandwidth changed by 20 %.

    ``boundary_bias`` flags estimation near the edge of the data. For a
    variable bounded below -- a duration, a count -- the symmetric
    kernel spills mass past the boundary and the density is biased
    downward there by roughly half.

    Parameters
    ----------
    x : array-like, shape (n,)
    at : array-like, optional
        Evaluation points; a 200-point grid by default.
    bandwidth : float, optional
    kernel : {'gaussian', 'epanechnikov', 'biweight', 'uniform',
              'triangular'}
    bw_method : {'silverman', 'scott'}

    Returns
    -------
    RichResult
        ``density``, ``at``, ``bandwidth``, ``bw_rule``,
        ``integral``, ``modes``, ``boundary_bias``.

    References
    ----------
    Silverman (1986), *Density Estimation for Statistics and Data
    Analysis*, Chapman and Hall, sections 3.3-3.4.
    Scott (1992). Epanechnikov (1969) for the optimal kernel.

    Examples
    --------
    >>> import numpy as np
    >>> out = kernel_density([0.0, 0.0, 0.0], at=[0.0])
    >>> bool(out["density"][0] > 0)
    True
    """
    v = np.asarray(x, dtype=float).ravel()
    n = v.size
    if n < 2:
        raise ValueError("need at least 2 observations, got %d." % n)
    if np.any(~np.isfinite(v)):
        raise ValueError("x contains non-finite values.")
    if kernel not in KERNELS:
        raise ValueError("kernel must be one of %s, got %r." % (KERNELS, kernel))
    if bw_method not in ("silverman", "scott"):
        raise ValueError(
            "bw_method must be 'silverman' or 'scott', got %r." % bw_method
        )
    sd = float(np.std(v, ddof=1))
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    if bandwidth is None:
        if bw_method == "silverman":
            a = min(sd, iqr / 1.34) if iqr > 0 else sd
            h = 0.9 * a * n ** (-0.2)
        else:
            h = 1.06 * sd * n ** (-0.2)
        h = float(max(h, 1e-9))
        auto = True
    else:
        h = float(bandwidth)
        auto = False
    if h <= 0:
        raise ValueError("bandwidth must be positive, got %r." % bandwidth)

    lo, hi = v.min() - 3 * h, v.max() + 3 * h
    grid = np.linspace(lo, hi, 200) if at is None else np.asarray(
        at, dtype=float
    ).ravel()
    dens = _k((grid[:, None] - v[None, :]) / h, kernel).sum(axis=1) / (n * h)
    integral = float(np.trapezoid(dens, grid)) if grid.size > 1 else np.nan

    modes = []
    if dens.size > 2:
        for i in range(1, dens.size - 1):
            if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]:
                modes.append(float(grid[i]))
    near = (grid < v.min() + h) | (grid > v.max() - h)
    return RichResult(
        payload={
            "estimate": dens,
            "density": dens,
            "at": grid,
            "bandwidth": h,
            "bandwidth_auto": auto,
            "bw_rule": bw_method if auto else "supplied",
            "bw_note": (
                "Silverman's rule is MISE-optimal for a GAUSSIAN target and "
                "oversmooths genuinely multimodal data -- exactly the case a "
                "density estimate was made for"
            ),
            "kernel": kernel,
            "kernel_note": (
                "Epanechnikov is MISE-optimal but a Gaussian kernel is about "
                "95 % as efficient; the kernel matters far less than the "
                "bandwidth"
            ),
            "integral": integral,
            "integral_note": (
                "should be close to 1 over a grid covering the support; a "
                "shortfall means the grid is truncating the tails"
            ),
            "modes": np.asarray(modes),
            "n_modes": int(len(modes)),
            "boundary_bias": near,
            "boundary_note": (
                "within one bandwidth of the data range a symmetric kernel "
                "spills mass outside the support and biases the density "
                "downward -- by roughly half at a hard boundary"
            ),
            "n": int(n),
            "method": "Kernel density estimate (%s)" % kernel,
        }
    )


def cheatsheet():
    return (
        "kerd: univariate KDE with the bandwidth rule surfaced and the "
        "boundary-bias region flagged"
    )
