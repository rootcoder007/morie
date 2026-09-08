# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explained variance ratio of the principal components."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["geron_explained_variance_ratio"]

_METHOD = "Explained variance ratio"


def geron_explained_variance_ratio(singular_values, threshold=0.95):
    r"""Share of variance carried by each principal component.

    .. math::
        \mathrm{EVR}_k = \frac{\sigma_k^2}{\sum_j \sigma_j^2}

    Squared singular values, not the singular values themselves --
    variance is a squared quantity, and using the raw values is the
    standard way to overstate the tail components.

    ``n_components_for_threshold`` answers the practical question the
    ratio exists for: how many components to keep for a given share of
    the variance.

    Parameters
    ----------
    singular_values : array-like, shape (k,)
        Non-negative singular values of the centred data matrix.
    threshold : float, optional
        Cumulative share to reach, in ``(0, 1]``. Default 0.95.

    Returns
    -------
    RichResult
        Payload keys ``explained_variance_ratio``, ``cumulative``,
        ``n_components_for_threshold``, ``total_variance``,
        ``estimate``, ``n``, ``method``.

    References
    ----------
    Géron Ch 7, Explained Variance Ratio section.

    Examples
    --------
    Singular values 3 and 4 give squares 9 and 16 out of 25:

    >>> r = geron_explained_variance_ratio([4.0, 3.0])
    >>> r["explained_variance_ratio"]
    [0.64, 0.36]
    >>> r["cumulative"]
    [0.64, 1.0]
    >>> r["n_components_for_threshold"]
    2

    Equal singular values split the variance evenly, so nothing can be
    dropped cheaply:

    >>> geron_explained_variance_ratio([1.0, 1.0, 1.0, 1.0],
    ...                                threshold=0.5)["n_components_for_threshold"]
    2
    """
    s = np.asarray(singular_values, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("singular_values is empty.")
    if not np.all(np.isfinite(s)):
        raise ValueError("singular_values must be finite.")
    if np.any(s < 0):
        raise ValueError("singular values are non-negative by definition; got a negative entry.")
    total = float(np.sum(s**2))
    if total == 0:
        raise ValueError("all singular values are zero: the data has no variance to explain.")
    threshold = float(threshold)
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must lie in (0, 1], got {threshold}.")

    evr = (s**2) / total
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, threshold - 1e-12) + 1)

    return RichResult(
        title="Explained variance ratio",
        summary_lines=[("Components", int(s.size)),
                       (f"Needed for {threshold:.0%}", k)],
        payload={
            "explained_variance_ratio": evr.tolist(),
            "cumulative": cum.tolist(),
            "n_components_for_threshold": k,
            "total_variance": total,
            "threshold": threshold,
            "estimate": evr.tolist(),
            "n": int(s.size),
            "method": _METHOD,
        },
    )


def cheatsheet():
    return "grevr: EVR_k = sigma_k^2 / sum_j sigma_j^2, plus components needed for a threshold"
