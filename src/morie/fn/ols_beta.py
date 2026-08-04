"""OLS coefficients of the response-surface model.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["ols_beta"]


def ols_beta(x, z):
    """OLS coefficients of the response-surface model

    Formula: beta_hat_OLS = (X^T X)^-1 X^T z

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (20.2).
    """
    arr = np.asarray(_brus.ols_beta(x, z), dtype=float)
    value = float(arr.ravel()[0])
    payload = {"values": arr.tolist(), "value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (20.2)"
    return RichResult(
        title='OLS coefficients of the response-surface model',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r20e2: beta_hat_OLS = (X^T X)^-1 X^T z [Brus 2022, eq. 20.2]'


# compact alias per ledger/NAMING.md
olsbeta = ols_beta
