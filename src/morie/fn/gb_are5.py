# morie.fn -- function file (rootcoder007/morie)
"""ARE of the scale tests against the F test."""

from ._gb_are import ARE_KLOTZ_VS_F_NORMAL, ARE_MOOD_VS_F_NORMAL
from ._richresult import RichResult

__all__ = ["gibbons_are_scale_tests"]


def gibbons_are_scale_tests(distribution="normal", cdf=None):
    r"""Scale-problem efficiencies at the normal (Gibbons
    Sec. 13.3.3, PDF-verified):

    .. math:: \mathrm{ARE}(M_N, F) = \frac{15}{2\pi^2} \approx 0.760,

    for Mood's squared-rank test, and ARE = 1 for the Klotz
    normal-scores test (Klotz 1962), which attains full efficiency
    where the F test is optimal.

    NOTE: the placeholder this module replaces claimed
    ARE(Mood, F) = 3/pi; the book's own derivation (Sec. 13.3.3, the
    e(M_N) calculation) gives 15/(2 pi^2). The placeholder value was
    wrong and is gone.

    Parameters
    ----------
    distribution : str
        Only "normal" is tabulated here; anything else raises rather
        than guessing.
    cdf : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``are_mood_f``, ``are_klotz_f``, ``distribution``,
        ``method``.

    References
    ----------
    Gibbons, J. D. & Chakraborti, S. (2021). *Nonparametric
    Statistical Inference* (5th ed.). CRC Press. Sec. 13.3.3.

    Klotz, J. (1962). Nonparametric tests for scale. *The Annals of
    Mathematical Statistics*, 33(2), 498-512.
    """
    if distribution != "normal":
        raise ValueError(
            "scale-test AREs are tabulated here for the normal only; "
            f"got {distribution!r}."
        )
    return RichResult(
        payload={
            "are_mood_f": ARE_MOOD_VS_F_NORMAL,
            "are_klotz_f": ARE_KLOTZ_VS_F_NORMAL,
            "distribution": "normal",
            "method": "ARE(Mood, F) = 15/(2 pi^2); ARE(Klotz, F) = 1 (Sec. 13.3.3)",
        }
    )


def cheatsheet():
    return "gb_are5: Mood 15/(2pi^2) = 0.760, Klotz 1.0 at the normal"
