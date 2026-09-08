"""Mixed-model calibration estimator.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np

from . import _brus
from ._richresult import RichResult

__all__ = ["mixed_calibration_mean"]


def mixed_calibration_mean(zbar_pi, a_hat, pi_sample, m_all_mean, m_ht_mean, b_hat, n_population):
    """Mixed-model calibration estimator

    Formula: zbar_MC = zbar_pi + a_hat(1 - (1/N) sum 1/pi) + b_hat(mbar - mbar_HT)

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Brus, D. J. (2022). Spatial Sampling with R. The R Series, CRC Press. Open-access edition: dickbrus.github.io/SpatialSamplingwithR,
    eq. (10.36).
    """
    value = _brus.mixed_calibration_mean(zbar_pi, a_hat, pi_sample, m_all_mean, m_ht_mean, b_hat, n_population)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Brus (2022) eq. (10.36)"
    return RichResult(
        title='Mixed-model calibration estimator',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return 'r10e36: zbar_MC = zbar_pi + a_hat(1 - (1/N) sum 1/pi) + b_hat(mbar - mbar_HT) [Brus 2022, eq. 10.36]'
