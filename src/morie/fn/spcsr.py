"""Complete spatial randomness (CSR): the homogeneous Poisson process."""

import numpy as np

from ._richresult import RichResult
from ._schab_pp import (as_points, as_region, intensity, nn_distances,
                        region_area)

__all__ = ["schabenberger_csr_def"]


def schabenberger_csr_def(points, region=None):
    r"""
    Complete spatial randomness, and a quadrat check of it.

    CSR is the homogeneous Poisson process: intensity constant over the
    region and events independent of one another, so there is neither
    attraction nor inhibition. It is the null every point-pattern test in
    Ch. 3 is stated against.

    Two consequences are reported, both of which a departure will break:

    * the quadrat counts have variance equal to their mean, so the
      index of dispersion :math:`s^2/\bar{x}` is about 1. Clustering
      pushes it ABOVE 1, regularity below.
    * the mean nearest-neighbour distance is
      :math:`1/(2\sqrt{\lambda})`; the ratio of observed to expected
      (Clark-Evans) is about 1 under CSR, below 1 when clustered and
      above 1 when regular.

    Parameters
    ----------
    points : array-like
        Event coordinates, shape ``(n, 2)``.
    region : array-like, optional
        ``(xmin, ymin, xmax, ymax)`` or vertices; bounding box of
        ``points`` when omitted.

    Returns
    -------
    RichResult
        ``index_of_dispersion``, ``quadrat_counts``, ``mean_nn``,
        ``expected_nn``, ``clark_evans``, ``lambda_est``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC. Ch. 3, Secs. 3.2-3.3.
    """
    p = as_points(points)
    reg = as_region(region, p)
    lam = intensity(p, reg)
    k = max(2, int(np.sqrt(p.shape[0] / 5.0)))          # ~5 events per quadrat
    xe = np.linspace(reg[0], reg[2], k + 1)
    ye = np.linspace(reg[1], reg[3], k + 1)
    counts, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=[xe, ye])
    counts = counts.ravel()
    mean_c = counts.mean()
    iod = float(counts.var(ddof=1) / mean_c) if mean_c > 0 else float("nan")

    nn = nn_distances(p)
    mean_nn = float(nn.mean()) if nn.size else float("nan")
    expected_nn = 1.0 / (2.0 * np.sqrt(lam)) if lam > 0 else float("nan")
    return RichResult(
        title="Complete spatial randomness",
        summary_lines=[("index of dispersion", iod),
                       ("Clark-Evans ratio", mean_nn / expected_nn
                        if expected_nn else float("nan"))],
        payload={"index_of_dispersion": iod, "quadrat_counts": counts,
                 "n_quadrats": int(counts.size), "mean_nn": mean_nn,
                 "expected_nn": float(expected_nn),
                 "clark_evans": float(mean_nn / expected_nn)
                 if expected_nn else float("nan"),
                 "lambda_est": lam, "area": region_area(reg)},
    )


def cheatsheet():
    return "spcsr: CSR null; dispersion index ~1, Clark-Evans ~1."
