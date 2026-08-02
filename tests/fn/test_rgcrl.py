"""Tests for rgcrl.rangayyan_correlation_dimension.

Spec: Grassberger, P. & Procaccia, I. (1983). Measuring the strangeness of
strange attractors. Physica D 9(1-2):189-208.

NOT Rangayyan. The 2024 edition mentions "correlation dimension" exactly once,
as a citation inside a sentence, and contains no occurrence of "Grassberger",
"Procaccia" or "correlation sum" -- so the previous "Ch 7" citation pointed at
nothing.

No worked example exists in the library, so the checks below are a direct
re-derivation of the correlation sum plus the dimensions the estimator is
defined to recover for signals whose attractor dimension is known a priori.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.rgcrl import rangayyan_correlation_dimension


def test_correlation_sum_matches_the_definition():
    """C_hat(r) = 2/(M(M-1)) * sum_{i<j} theta(r - ||Y_i - Y_j||).

    theta is the Heaviside step with theta(u) = 1 for u >= 0, so the indicator
    is ||.|| <= r. Recomputed here from the definition against the log_C the
    function reports.
    """
    x = np.random.default_rng(31).standard_normal(120)
    m, tau = 3, 1
    res = rangayyan_correlation_dimension(x, m=m, tau=tau, n_r=15)

    M = x.size - (m - 1) * tau
    Y = np.column_stack([x[i * tau : i * tau + M] for i in range(m)])
    d = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    iu = np.triu_indices(M, k=1)
    dist = d[iu]

    # Rebuild the radii the way the function does rather than round-tripping
    # through exp(log r): the round trip loses the last bits, so a pair lying
    # exactly at a radius flips out of the count and the comparison fails for
    # a reason that has nothing to do with the estimator.
    pos = dist[dist > 0]
    rs = np.logspace(np.log10(max(pos.min(), 1e-12)), np.log10(dist.max()), 15)
    for r in rs:
        n_pairs = int(np.count_nonzero(dist <= r))
        if n_pairs < 10:
            continue                      # below the function's usable floor
        want = 2.0 * n_pairs / (M * (M - 1))
        idx = int(np.argmin(np.abs(res["log_r"] - np.log(r))))
        assert np.isclose(np.exp(res["log_C"][idx]), want, rtol=1e-12)


def test_identity_white_noise_fills_the_embedding_space():
    """Gaussian noise has no attractor, so D2 tracks the embedding dimension.

    The estimate is noisy and biased low at finite N, so this asserts the
    trend rather than a value: D2 must rise with m, not saturate.
    """
    x = np.random.default_rng(37).standard_normal(1500)
    d2 = [rangayyan_correlation_dimension(x, m=m, tau=1, n_r=25)["D2"]
          for m in (2, 4, 6)]
    assert d2[0] < d2[1] < d2[2], f"D2 should grow with m for noise, got {d2}"


def test_identity_sine_is_a_one_dimensional_curve():
    """A pure sine traces a closed curve in phase space, so D2 -> 1.

    This is the standard sanity check for the estimator: a periodic orbit is
    a 1-D manifold however high the embedding dimension.
    """
    t = np.linspace(0, 40 * np.pi, 1200)
    x = np.sin(t)
    d2 = rangayyan_correlation_dimension(x, m=4, tau=8, n_r=25)["D2"]
    assert 0.6 < d2 < 1.6, f"sine should give D2 near 1, got {d2}"


def test_identity_scale_invariance():
    """D2 is a dimension, so it cannot move under affine rescaling.

    Scaling x multiplies every pairwise distance by |a|, which shifts log r by
    a constant and leaves the slope untouched; a translation cancels in the
    differences.
    """
    x = np.random.default_rng(41).standard_normal(400)
    base = rangayyan_correlation_dimension(x, m=3, tau=1, n_r=20)["D2"]
    for a, b in ((100.0, 0.0), (0.01, 0.0), (1.0, 25.0), (-3.0, -8.0)):
        got = rangayyan_correlation_dimension(a * x + b, m=3, tau=1, n_r=20)["D2"]
        assert np.isclose(got, base, rtol=1e-9, atol=1e-9)


def test_rejects_array_where_a_scalar_belongs():
    """The generated test called this as f(x, y), putting a whole series where
    the embedding dimension goes.

    That surfaced as "truth value of an array with more than one element is
    ambiguous" from inside the embedding -- a message that says nothing about
    the actual mistake. The signature is (x, m, tau, n_r); passing an array
    for any of the three scalars now says so.
    """
    x = np.arange(50, dtype=float)
    with pytest.raises(ValueError, match="must be a scalar integer"):
        rangayyan_correlation_dimension(x, np.arange(50, dtype=float))
    with pytest.raises(ValueError, match="must be a scalar integer"):
        rangayyan_correlation_dimension(x, m=3, tau=np.array([1, 2]))


def test_rejects_series_too_short_to_embed():
    """The old edge test passed two samples and asserted result["n"] == 2 --
    a key this function does not return, from a computation it cannot do."""
    with pytest.raises(ValueError, match="Series too short"):
        rangayyan_correlation_dimension(np.array([1.0, 2.0]), m=3)


def test_returns_documented_keys():
    res = rangayyan_correlation_dimension(
        np.random.default_rng(43).standard_normal(200), m=3, tau=1, n_r=15)
    for key in ("D2", "log_r", "log_C", "m", "tau"):
        assert key in res
    assert np.isfinite(res["D2"])
