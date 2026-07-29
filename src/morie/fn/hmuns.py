# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unsupervised learning: discover structure from unlabeled data."""

import numpy as np

from ._richresult import RichResult
from .hmaen import geron_autoencoder
from .hmagc import geron_agglomerative
from .hmsil import geron_silhouette

__all__ = ["geron_unsupervised_learning"]


def geron_unsupervised_learning(X, n_clusters=2, bottleneck=1, linkage="average"):
    """
    Unsupervised learning: discover structure from unlabeled data.

    Formula: maximize p_theta(X) or cluster assignments

    Runs the two things "structure without labels" actually means and
    scores both, delegating each to the module that owns it rather than
    reimplementing anything:

    * **grouping** -- :func:`morie.fn.hmagc.geron_agglomerative`, scored
      by :func:`morie.fn.hmsil.geron_silhouette` (there are no labels, so
      the score has to be internal);
    * **compression** -- :func:`morie.fn.hmaen.geron_autoencoder`, whose
      reconstruction error and explained-variance ratio say how much of
      the data a low-dimensional code keeps.

    Together they answer the only two questions available without labels:
    does the data fall into groups, and does it live on a lower-dimensional
    surface.

    Parameters
    ----------
    X : array-like
        Unlabeled data (n, d), n >= 2.
    n_clusters : int, default 2
        Groups to look for (2 <= n_clusters < n).
    bottleneck : int, default 1
        Code width for the compression view (1 <= bottleneck <= d).
    linkage : str, default "average"
        Linkage rule passed to the clusterer.

    Returns
    -------
    result : RichResult
        Keys: labels, silhouette, codes, recon_error,
        explained_variance_ratio, estimate, n, method.

    Examples
    --------
    Two tight groups lying on a line: the clustering finds the groups and
    a single code unit reconstructs the data exactly.

    >>> X = [[0.0, 0.0], [0.2, 0.2], [5.0, 5.0], [5.2, 5.2]]
    >>> r = geron_unsupervised_learning(X, n_clusters=2, bottleneck=1)
    >>> [int(v) for v in r["labels"]]
    [0, 0, 1, 1]
    >>> bool(r["silhouette"] > 0.9)
    True
    >>> round(float(r["recon_error"]), 12)
    0.0
    >>> round(float(r["explained_variance_ratio"][0]), 12)
    1.0

    References
    ----------
    Géron Ch 1
    """
    A = np.asarray(X, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2 or A.shape[0] < 2:
        raise ValueError("geron_unsupervised_learning: X must be 2-D with at least 2 rows")
    if not np.all(np.isfinite(A)):
        raise ValueError("geron_unsupervised_learning: X contains non-finite values")
    k = int(n_clusters)
    if not (2 <= k < A.shape[0]):
        raise ValueError(
            f"geron_unsupervised_learning: n_clusters must lie in 2..{A.shape[0] - 1}, got {k}"
        )
    bn = int(bottleneck)
    if not (1 <= bn <= A.shape[1]):
        raise ValueError(f"geron_unsupervised_learning: bottleneck must lie in 1..{A.shape[1]}, got {bn}")

    grp = geron_agglomerative(A, n_clusters=k, linkage=linkage)
    labels = np.asarray(grp["labels"]).ravel()
    sil = geron_silhouette(A, labels)
    comp = geron_autoencoder(A, bn)

    return RichResult(
        title="Unsupervised structure discovery",
        summary_lines=[
            ("Points", int(A.shape[0])),
            ("Clusters", k),
            ("Mean silhouette", float(sil["silhouette"])),
            ("Reconstruction MSE at width " + str(bn), float(comp["recon_error"])),
        ],
        interpretation=(
            "Without labels there is no accuracy to report: cluster quality is judged internally "
            "(silhouette) and compression by how little is lost, and the two can disagree."
        ),
        payload={
            "labels": labels,
            "silhouette": float(sil["silhouette"]),
            "silhouette_samples": np.asarray(sil["samples"], dtype=float),
            "codes": np.asarray(comp["codes"], dtype=float),
            "reconstruction": np.asarray(comp["reconstruction"], dtype=float),
            "recon_error": float(comp["recon_error"]),
            "explained_variance_ratio": np.asarray(comp["explained_variance_ratio"], dtype=float),
            "merge_heights": grp["heights"],
            "estimate": float(sil["silhouette"]),
            "n": int(A.shape[0]),
            "method": "Unsupervised structure: agglomerative grouping (hmagc) + silhouette (hmsil) + linear autoencoder (hmaen)",
        },
    )


def cheatsheet():
    return "hmuns: Unsupervised learning: discover structure from unlabeled data"
