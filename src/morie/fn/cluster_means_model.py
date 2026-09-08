"""Per-cluster level-1 models y_ij = beta0j + e_ij.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["cluster_means_model"]


def cluster_means_model(groups):
    """Per-cluster level-1 models y_ij = beta0j + e_ij

    Formula: y_i1 = beta0,1 + e_i1; ...; y_ik = beta0,k + e_ik

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'grand_mean' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.4
    """
    payload = dict(_ca_crim.cluster_means_model(groups))
    value = payload['grand_mean']
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.4)"
    return RichResult(
        title='Per-cluster level-1 models y_ij = beta0j + e_ij',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e4: y_i1 = beta0,1 + e_i1; ...; y_ik = beta0,k + e_ik [Weisburd et al. 2022, eq. 7.4]'
