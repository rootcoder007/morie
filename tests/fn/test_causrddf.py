"""Tests for causrddf.causal_rdd_fuzzy."""

from morie.fn import _array_core as np

from morie.fn.causrddf import causal_rdd_fuzzy


def test_causrddf_basic():
    """Test basic functionality.

    Build a fuzzy RDD design: the running variable x is uniform on a
    wide range, the outcome y has a discontinuity at the cutoff, and
    the treatment probability treat jumps from 0 (below cutoff) to 1
    (above cutoff) with noise -- so the first-stage jump is non-zero
    and the function should run cleanly.
    """
    rng = np.random.default_rng(42)
    n = 400
    x = rng.uniform(-5.0, 5.0, n)
    # Outcome: smooth in x with a positive jump at the cutoff.
    y = 0.5 * x + 1.0 + (x >= 0.0) * 1.0 + rng.normal(0, 0.2, n)
    # Treatment: near-deterministic step function (fuzzy = not perfectly sharp).
    treat = ((x >= 0.0).astype(float) + rng.normal(0, 0.05, n))
    treat = np.clip(treat, 0.0, 1.0)

    cutoff = 0.0
    h = 2.0
    result = causal_rdd_fuzzy(x, y, treat, cutoff, h)
    assert isinstance(result, dict)
    # Documented keys in the RichResult payload.
    for key in ("estimate", "se", "ci",
                "jump_outcome", "jump_treatment",
                "se_outcome", "se_treatment",
                "h_outcome", "h_treatment"):
        assert key in result, f"missing key {key!r} in result"

    # Point estimate is the documented ratio of the two jumps.
    expected_tau = result["jump_outcome"] / result["jump_treatment"]
    assert result["estimate"] == expected_tau

    # Delta-method SE using the documented formula
    # se^2 = (se_outcome^2 + tau^2 * se_treatment^2) / jump_treatment^2.
    tau = result["estimate"]
    expected_se = np.sqrt(
        (result["se_outcome"] ** 2
         + tau ** 2 * result["se_treatment"] ** 2)
        / result["jump_treatment"] ** 2
    )
    assert result["se"] == expected_se


def test_causrddf_edge():
    """Test edge cases: bandwidths may be passed positionally or by name.

    Same DGP as the basic test but exercised with a small bandwidth on
    each side, and a fresh RNG seed, to make sure the documented
    `h` and `h_treat` keyword arguments are both accepted.
    """
    rng = np.random.default_rng(42)
    n = 400
    x = rng.uniform(-5.0, 5.0, n)
    y = 0.5 * x + 1.0 + (x >= 0.0) * 1.0 + rng.normal(0, 0.2, n)
    treat = np.clip(
        (x >= 0.0).astype(float) + rng.normal(0, 0.05, n),
        0.0, 1.0,
    )
    cutoff = 0.0
    h = 2.0
    result = causal_rdd_fuzzy(x, y, treat, cutoff=cutoff, h=h, h_treat=h)
    assert isinstance(result, dict)
    assert result["h_outcome"] == h
    assert result["h_treatment"] == h
