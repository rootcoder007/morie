# morie.fn -- function file (rootcoder007/morie)
"""Kernel intensity estimation for an inhomogeneous Poisson process."""

import numpy as np

from ._richresult import RichResult

__all__ = ["schabenberger_intensity_estimation"]

KERNELS = ("gaussian", "quadratic", "minimum_variance", "uniform")


def _kernel(t, kind):
    """Kernel functions of Schabenberger p. 111.

    Any k with unit integral and zero first moment will serve; the book
    is explicit that the choice of kernel matters far less in practice
    than the choice of bandwidth.
    """
    t = np.asarray(t, dtype=float)
    if kind == "gaussian":
        return np.exp(-0.5 * t ** 2) / np.sqrt(2.0 * np.pi)
    if kind == "quadratic":
        return np.where(np.abs(t) <= 1, 0.75 * (1 - t ** 2), 0.0)
    if kind == "minimum_variance":
        return np.where(np.abs(t) <= 1, (3.0 / 8.0) * (3 - 5 * t ** 2), 0.0)
    if kind == "uniform":
        return np.where(np.abs(t) <= 1, 0.5, 0.0)
    raise ValueError("kernel must be one of %s, got %r." % (KERNELS, kind))


def schabenberger_intensity_estimation(points, bandwidth=None, region=None,
                                       grid=40, kernel="gaussian",
                                       edge_correct=True):
    r"""Kernel estimate of a first-order intensity, Schabenberger eq (3.14).

    The product-kernel estimator is

    .. math::
       \hat\lambda(s_0) = \frac{1}{\nu(A) h_x h_y}
       \sum_{i=1}^{n} k\!\left(\frac{x_i - x_0}{h_x}\right)
                      k\!\left(\frac{y_i - y_0}{h_y}\right),

    and Diggle's edge-corrected form with a single bandwidth is

    .. math::
       \hat\lambda(s) = \frac{1}{p_h(s)}\sum_{i=1}^{n}
         \frac{1}{h^{2}}k\!\left(\frac{s - s_i}{h}\right),
       \qquad
       p_h(s) = \int_A h^{-2}k\!\left(\frac{s-u}{h}\right)du .

    Edge effects are not a detail. Near the boundary of :math:`A` part
    of every kernel falls outside the region, so an uncorrected
    estimate is biased downward exactly where boundary artefacts are
    easiest to over-interpret; :math:`p_h(s)` is the fraction of kernel
    mass that stayed inside, and dividing by it restores the scale.

    Intensity and density differ only by a constant:
    :math:`\lambda(s) = f_A(s)\,\mu(A)`, so this is density estimation
    with the total count carried through rather than divided out.

    The bandwidth is the choice that matters. Small :math:`h` gives a
    nearly unbiased but highly variable surface, large :math:`h` a
    smooth and badly biased one, and the book's own worked example
    reports that automatic selection produced a map "far too smooth"
    to be useful. The default here is Silverman's rule, reported in
    ``bandwidth`` so it is visible as a choice rather than a fact.

    Parameters
    ----------
    points : array-like, shape (n, 2)
        Event locations.
    bandwidth : float or (float, float), optional
        Single bandwidth, or ``(hx, hy)`` for the product kernel.
    region : (xmin, xmax, ymin, ymax), optional
        Study region A. The bounding box of the points by default,
        which UNDERSTATES the true region and so overstates the
        intensity; pass the real one when it is known.
    grid : int
        Evaluation grid resolution per axis.
    kernel : {'gaussian', 'quadratic', 'minimum_variance', 'uniform'}
    edge_correct : bool
        Apply Diggle's correction.

    Returns
    -------
    RichResult
        ``intensity_surface`` (grid by grid), ``x_grid``, ``y_grid``,
        ``bandwidth``, ``area``, ``mean_intensity``,
        ``integrated_intensity``, ``edge_weight_min``.

    References
    ----------
    Schabenberger and Gotway (2005), section 3.5.1, equations
    (3.12)-(3.14) and Diggle's edge-corrected estimator, pp. 110-113.
    Diggle (1985), *Applied Statistics* 34:138-147.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> pts = rng.uniform(0, 10, size=(200, 2))
    >>> out = schabenberger_intensity_estimation(pts, region=(0, 10, 0, 10))
    >>> bool(abs(out["mean_intensity"] - 2.0) < 1.0)
    True
    """
    P = np.atleast_2d(np.asarray(points, dtype=float))
    if P.shape[1] != 2:
        P = P.T
    if P.shape[1] != 2:
        raise ValueError("points must have two coordinate columns.")
    n = P.shape[0]
    if n < 2:
        raise ValueError("need at least 2 events, got %d." % n)
    if kernel not in KERNELS:
        raise ValueError("kernel must be one of %s, got %r." % (KERNELS, kernel))

    if region is None:
        xmin, xmax = float(P[:, 0].min()), float(P[:, 0].max())
        ymin, ymax = float(P[:, 1].min()), float(P[:, 1].max())
        region_given = False
    else:
        xmin, xmax, ymin, ymax = (float(v) for v in region)
        region_given = True
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("region must have positive width and height.")
    area = (xmax - xmin) * (ymax - ymin)

    if bandwidth is None:
        sd = np.sqrt(0.5 * (np.var(P[:, 0]) + np.var(P[:, 1])))
        h = float(1.06 * sd * n ** (-1.0 / 6.0))
        h = max(h, 1e-9)
        hx = hy = h
        auto = True
    else:
        b = np.atleast_1d(np.asarray(bandwidth, dtype=float))
        hx, hy = (float(b[0]), float(b[0])) if b.size == 1 else (
            float(b[0]), float(b[1]))
        auto = False
    if hx <= 0 or hy <= 0:
        raise ValueError("bandwidth must be positive.")

    g = int(grid)
    xs = np.linspace(xmin, xmax, g)
    ys = np.linspace(ymin, ymax, g)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    lam = np.zeros((g, g))
    for i in range(n):
        lam += (_kernel((GX - P[i, 0]) / hx, kernel)
                * _kernel((GY - P[i, 1]) / hy, kernel))
    lam /= (hx * hy)

    # Diggle's p_h(s): the kernel mass remaining inside A, evaluated by
    # the same grid quadrature used everywhere else so the correction
    # and the surface are consistent
    edge = np.ones((g, g))
    if edge_correct:
        dx = (xmax - xmin) / max(g - 1, 1)
        dy = (ymax - ymin) / max(g - 1, 1)
        for a in range(g):
            for b_ in range(g):
                w = (_kernel((GX - xs[a]) / hx, kernel)
                     * _kernel((GY - ys[b_]) / hy, kernel)) / (hx * hy)
                edge[a, b_] = float(np.sum(w) * dx * dy)
        edge = np.maximum(edge, 1e-6)
        lam = lam / edge

    dx = (xmax - xmin) / max(g - 1, 1)
    dy = (ymax - ymin) / max(g - 1, 1)
    integrated = float(np.sum(lam) * dx * dy)
    return RichResult(
        payload={
            "estimate": lam,
            "intensity_surface": lam,
            "x_grid": xs,
            "y_grid": ys,
            "bandwidth": (hx, hy),
            "bandwidth_auto": auto,
            "bandwidth_note": (
                "the bandwidth, not the kernel, is what determines the "
                "surface; a small one is nearly unbiased but unstable, a "
                "large one smooth and biased"
            ),
            "kernel": kernel,
            "edge_corrected": bool(edge_correct),
            "edge_weight": edge,
            "edge_weight_min": float(edge.min()),
            "edge_note": (
                "p_h(s) is the fraction of kernel mass falling inside A; "
                "without dividing by it the intensity is biased downward at "
                "the boundary, exactly where artefacts are easiest to "
                "over-read"
            ),
            "area": area,
            "region_supplied": region_given,
            "region_note": (
                None if region_given else
                "no region was given, so A is the bounding box of the events "
                "-- which is smaller than the true study region and "
                "therefore overstates the intensity"
            ),
            "mean_intensity": float(n / area),
            "integrated_intensity": integrated,
            "n": n,
            "method": "Kernel intensity estimation for an inhomogeneous "
                      "Poisson process",
        }
    )


def cheatsheet():
    return (
        "spintp: product-kernel first-order intensity (3.14) with Diggle's "
        "edge correction, and the bandwidth surfaced as the real choice"
    )
