"""Wald confidence interval for pi via the logit scale.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["pi_wald_interval"]


def pi_wald_interval(xb, var_xb, z):
    """Wald confidence interval for pi via the logit scale

    Formula: expit(Xb +/- z sqrt(Var(Xb)))

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'pi' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (2.15).
    """
    payload = dict(_acd.pi_wald_interval(xb, var_xb, z))
    value = float(payload['pi'])
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (2.15)"
    return RichResult(
        title='Wald confidence interval for pi via the logit scale',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '2e15: expit(Xb +/- z sqrt(Var(Xb))) [Bilder & Loughin 2025, eq. 2.15]'


# compact alias per ledger/NAMING.md
piwaldinterval = pi_wald_interval
