"""Variance component sigma2_u = (MSbetween - MSwithin)/n.

Book-as-spec implementation; see reference for context.
"""

import math as _math  # noqa: F401

from . import _ca_crim
from ._richresult import RichResult

__all__ = ["ca_chapter_7_equation_6"]


def ca_chapter_7_equation_6(ms_between, ms_within, n_per_cluster):
    """Variance component sigma2_u = (MSbetween - MSwithin)/n

    Formula: sigma^2_u = (MS_between - MS_within) / n

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Weisburd, Wilson, Wooditch & Britt (2022). Advanced Statistics in Criminology and Criminal Justice, 5th ed. Springer. doi:10.1007/978-3-030-67738-1,
    ch.7 eq.7.6
    """
    value = _ca_crim.variance_components_sigma2_u(ms_between, ms_within, n_per_cluster)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Weisburd et al. (2022) eq. (7.6)"
    return RichResult(
        title='Variance component sigma2_u = (MSbetween - MSwithin)/n',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'ca7e6: sigma^2_u = (MS_between - MS_within) / n [Weisburd et al. 2022, eq. 7.6]'
