"""Expected number of tests in Dorfman group testing.

Book-as-spec implementation; see reference for context.
"""

from . import _array_core as np  # noqa: F401

from . import _acd
from ._richresult import RichResult

__all__ = ["group_testing_expected_tests"]


def group_testing_expected_tests(i_size, se, sp, pi_tilde):
    """Expected number of tests in Dorfman group testing

    Formula: E(T_k) = 1 + I [Se + (1 - Se - Sp)(1 - pi_tilde)^I]

    Returns
    -------
    result : RichResult
        dict subclass; headline key 'value' plus the full payload.

    References
    ----------
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R, 2nd ed. Chapman & Hall/CRC,
    eq. (6.26).
    """
    value = _acd.group_testing_expected_tests(i_size, se, sp, pi_tilde)
    payload = {"value": value}
    summary = [(k, v) for k, v in payload.items()
               if isinstance(v, (int, float))][:4]
    payload = dict(payload)
    payload.setdefault("value", value)
    payload["method"] = "Bilder & Loughin (2025) eq. (6.26)"
    return RichResult(
        title='Expected number of tests in Dorfman group testing',
        summary_lines=summary,
        payload=payload,
    )


def cheatsheet():
    return '6e26: E(T_k) = 1 + I [Se + (1 - Se - Sp)(1 - pi_tilde)^I] [Bilder & Loughin 2025, eq. 6.26]'
