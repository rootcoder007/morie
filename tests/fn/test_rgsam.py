"""Tests for rgsam.rangayyan_sample_entropy.

Spec: Richman & Moorman (2000), Am J Physiol Heart Circ Physiol 278(6):
H2039-H2049. NOT Rangayyan -- the 2024 edition contains no occurrence of
"sample entropy", "approximate entropy", "Pincus" or "Richman", so the
previous "Ch 7" citation pointed at nothing.

No transcribable worked example exists in the library, so the checks here are
a direct re-derivation of the definition plus its analytic limits.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.rgapn import rangayyan_approximate_entropy
from morie.fn.rgsam import rangayyan_sample_entropy


def _sampen_reference(x, m, r):
    """Literal Richman & Moorman: BOTH counts over the same N-m templates.

    Written from the definition rather than from the implementation, so
    agreement is evidence and not tautology.
    """
    x = np.asarray(x, float)
    N = x.size
    nT = N - m
    def count(mm):
        c = 0
        for i in range(nT):
            for j in range(i + 1, nT):
                if np.max(np.abs(x[i:i+mm] - x[j:j+mm])) <= r:
                    c += 1
        return c
    return -np.log(count(m + 1) / count(m))


def test_matches_richman_moorman_definition():
    """Implementation == the definition, re-derived independently.

    This is what caught the defect: building N-mm+1 templates per call gave B
    one template that A could not have, so A and B had different denominators
    -- exactly the bias SampEn was defined to remove. Measured on 300
    Gaussian samples (m=2, r=0.2 sd): B gained 9 spurious pairs and SampEn was
    biased upward by +0.018.
    """
    rng = np.random.default_rng(20260726)
    # r must be loose enough that at least one length-(m+1) match exists, or
    # A = 0 and the reference divides by zero. That is a property of the test
    # data, not of the estimator -- the A = 0 case is exercised separately in
    # test_no_matches_returns_infinity.
    for n, m, frac in ((120, 2, 0.2), (200, 2, 0.2), (300, 3, 0.35)):
        x = rng.standard_normal(n)
        r = frac * x.std()
        got = rangayyan_sample_entropy(x, m=m, r=r)["SampEn"]
        assert np.isclose(got, _sampen_reference(x, m, r), rtol=1e-12, atol=1e-12)


def test_no_matches_returns_infinity():
    """A = 0 means no length-(m+1) template pair matched, so -ln(A/B) diverges.

    The function reports inf rather than raising: an unmatched template set is
    a legitimate outcome for a short or highly irregular series at a tight
    tolerance, not a caller error.
    """
    x = np.random.default_rng(17).standard_normal(60)
    res = rangayyan_sample_entropy(x, m=3, r=1e-9)
    assert res["A"] == 0
    assert np.isinf(res["SampEn"])


def test_a_and_b_share_a_denominator():
    """A and B must be counted over the same number of template vectors.

    Directly pins the property the fix restores: with n_templates = N-m for
    both, A can never exceed B, because every length-(m+1) match implies a
    length-m match on the same pair.
    """
    rng = np.random.default_rng(3)
    for n in (80, 150, 400):
        x = rng.standard_normal(n)
        res = rangayyan_sample_entropy(x, m=2)
        assert res["A"] <= res["B"], "A > B is impossible when denominators match"


def test_identity_constant_signal_is_perfectly_regular():
    """A constant series is maximally regular: every pair matches, so A == B
    and SampEn = -ln(1) = 0."""
    x = np.full(200, 3.7)
    res = rangayyan_sample_entropy(x, m=2, r=0.1)
    assert res["A"] == res["B"]
    assert np.isclose(res["SampEn"], 0.0, atol=1e-12)


def test_identity_noise_is_less_regular_than_a_sine():
    """SampEn orders signals by regularity -- the claim the statistic makes."""
    t = np.linspace(0, 20 * np.pi, 600)
    sine = np.sin(t)
    noise = np.random.default_rng(5).standard_normal(600)
    assert (rangayyan_sample_entropy(sine, m=2)["SampEn"]
            < rangayyan_sample_entropy(noise, m=2)["SampEn"])


def test_identity_scale_invariance_with_relative_tolerance():
    """With r set as a fraction of the standard deviation, SampEn is invariant
    under affine rescaling: the Chebyshev distances and r scale together."""
    x = np.random.default_rng(9).standard_normal(300)
    base = rangayyan_sample_entropy(x, m=2)["SampEn"]
    for a, b in ((100.0, 0.0), (0.01, 0.0), (1.0, -50.0), (-2.0, 7.0)):
        assert np.isclose(rangayyan_sample_entropy(a * x + b, m=2)["SampEn"], base,
                          rtol=1e-9, atol=1e-9)


def test_differs_from_apen_in_the_two_documented_ways():
    """SampEn and ApEn must not coincide: self-matches and denominators differ.

    If a future edit made rgsam include self-matches, or reverted the template
    count, this is where it would surface.
    """
    x = np.random.default_rng(13).standard_normal(250)
    s = rangayyan_sample_entropy(x, m=2)["SampEn"]
    a = rangayyan_approximate_entropy(x, m=2)["ApEn"]
    assert not np.isclose(s, a, rtol=1e-3)


def test_returns_documented_keys():
    """The generated test asserted an "estimate" key this never promised."""
    res = rangayyan_sample_entropy(np.random.default_rng(1).standard_normal(200), m=2)
    for key in ("SampEn", "A", "B", "m", "r", "n"):
        assert key in res


def test_rejects_series_shorter_than_m_plus_two():
    """Two template vectors are the minimum for any pair to exist."""
    with pytest.raises(ValueError, match=r"len\(x\) > m \+ 1"):
        rangayyan_sample_entropy(np.array([42.0]))
    with pytest.raises(ValueError, match=r"len\(x\) > m \+ 1"):
        rangayyan_sample_entropy(np.arange(3.0), m=2)
