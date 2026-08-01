"""Between/within decomposition with the cluster mean.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_11"]


def ca_chapter_7_equation_11(b0, b1, b2, x_ij, cluster_mean, u_j):
    """Between/within decomposition with the cluster mean

    Formula: y_ij = beta0 + beta1 (x_1ij - xbar_1j) + beta2 xbar_1j + u_j + e_ij

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.11
    """
    value = _ca_crim.multilevel_predict(b0, [b1, b2], [x_ij - cluster_mean, cluster_mean], [u_j])
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.11)"
    return RichResult(
        title='Between/within decomposition with the cluster mean',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e11: y_ij = beta0 + beta1 (x_1ij - xbar_1j) + beta2 xbar_1j + u_j + e_ij [Weisburd et al. 2022, eq. 7.11]'
