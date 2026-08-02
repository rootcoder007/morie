"""Beta posterior for a binomial probability.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_23"]


def analysis_of_categorical_data_with_r_chapman_hall_crc_christo_chapter_6_equation_23(pi, w, n, a, b):
    """Beta posterior for a binomial probability

    Formula: p(pi|w) = Beta(w + a, n - w + b) density

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.23).
    """
    value = _acd.posterior_density_binomial(pi, w, n, a, b)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.23)"
    return RichResult(
        title='Beta posterior for a binomial probability',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e23: p(pi|w) = Beta(w + a, n - w + b) density [Bilder & Loughin 2025, eq. 6.23]'
