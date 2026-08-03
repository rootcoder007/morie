"""Cluster-sampling total, SI-of-clusters form.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["cluster_total_si"]


def cluster_total_si(cluster_totals, n_clusters_population, n):
    """Cluster-sampling total, SI-of-clusters form

    Formula: t_hat(z) = (N/n) sum t_j

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (6.9).
    """
    value = _brus.cluster_total_si(cluster_totals, n_clusters_population, n)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (6.9)"
    return RichResult(
        title='Cluster-sampling total, SI-of-clusters form',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r6e9: t_hat(z) = (N/n) sum t_j [Brus 2022, eq. 6.9]'
