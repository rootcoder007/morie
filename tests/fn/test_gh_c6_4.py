"""Tests for gh_c6_4.ghosal_df_inconsist."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c6_4 import ghosal_df_inconsist


def _compute_log_odds_independently(n, seed):
    """Independent re-implementation of the log_odds computation
    to derive a numeric expectation from first principles."""
    rng = np.random.default_rng(seed)
    phi_raw = [1.0 / (i * math.log(i + 1.0) ** 2) for i in range(2, 40)]
    tot = sum(phi_raw)
    phi = [v / tot for v in phi_raw]
    theta0 = phi[:]
    data = []
    for _ in range(n):
        u = float(rng.uniform(0, 1))
        acc = 0.0
        for i, p in enumerate(theta0):
            acc += p
            if u <= acc:
                data.append(i)
                break
        else:
            data.append(len(theta0) - 1)
    ll_phi = sum(math.log(phi[x]) for x in data)
    M_tot = sum(2.0 ** (-(i + 1)) for i in range(len(phi)))
    counts = {}
    ll_dp = 0.0
    for j, x in enumerate(data):
        a_x = 2.0 ** (-(x + 1))
        ll_dp += math.log((a_x + counts.get(x, 0)) / (M_tot + j))
        counts[x] = counts.get(x, 0) + 1
    return ll_phi - ll_dp, ll_phi, ll_dp


def test_gh_c6_4_basic():
    """Test basic functionality: signature is (n=400, seed=42),
    and result is a RichResult whose payload carries 'estimate'."""
    n, seed = 400, 42
    result = ghosal_df_inconsist(n=n, seed=seed)

    # Result must be a RichResult-like mapping; the actual estimate lives
    # inside the .payload attribute per the implementation.
    payload = getattr(result, "payload", result)

    assert "estimate" in payload
    estimate = payload["estimate"]
    assert np.all(np.isfinite(np.asarray(estimate, dtype=float)))

    # Independent reproduction of the log_odds using the same RNG seed
    # and the documented formula, with no reference to the function's output.
    expected_log_odds, expected_ll_phi, expected_ll_dp = _compute_log_odds_independently(n, seed)
    assert math.isfinite(expected_log_odds)

    # The formula is deterministic given the seed: the implementation and
    # the independent expression must agree exactly.
    assert estimate == expected_log_odds
    # Sanity: the two component log-likelihoods sum to the odds.
    assert expected_ll_phi - expected_ll_dp == expected_log_odds

    # Documentation says the delta component wins on this example.
    assert payload["delta_component_wins"] is True
    assert payload["delta_component_wins"] == (expected_log_odds > 0)
    assert payload["method"].startswith("DF inconsistency")


def test_gh_c6_4_edge():
    """Test edge cases: single draw still returns a finite log_odds estimate
    and the documented payload keys; there is no 'n' key."""
    result = ghosal_df_inconsist(n=1, seed=42)
    payload = getattr(result, "payload", result)

    assert "estimate" in payload
    assert np.all(np.isfinite(np.asarray(payload["estimate"], dtype=float)))
    assert "n" not in payload

    # Independent check for n=1
    expected, _, _ = _compute_log_odds_independently(1, 42)
    assert payload["estimate"] == expected
