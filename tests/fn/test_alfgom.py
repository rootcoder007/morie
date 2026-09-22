"""Tests for alfgom.alphago_montecarlo."""

from morie.fn import _array_core as np

from morie.fn.alfgom import alphago_montecarlo


def _uniforms(n, seed=0):
    """Generate n uniforms in [0, 1) using the shim's default_rng."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(float(rng.random()))
    return out


def test_alfgom_basic():
    """Test basic functionality with the documented formula."""
    # Per docstring: state is a scalar; rollout_net is s -> p;
    # step(s, a), terminal(s), outcome(s), value_net(s) are callables.
    state = 0
    horizon = 4

    # Simple 2-action rollout policy: equal probability [0.5, 0.5]
    def rollout_net(s):
        return [0.5, 0.5]

    # Simple environment: never terminal, always take action 0, identity state
    def step(s, a):
        return s + 1

    def terminal(s):
        return False

    def outcome(s):
        # Outcome equals the final state as a float value in [-1, 1]
        return 0.25

    def value_net(s):
        return -0.5

    lam = 0.5
    stream = _uniforms(horizon, seed=42)

    result = alphago_montecarlo(
        state,
        rollout_net,
        horizon=horizon,
        step=step,
        terminal=terminal,
        outcome=outcome,
        value_net=value_net,
        lam=lam,
        stream=stream,
    )

    # RichResult behaves like a dict; check the documented payload keys.
    assert isinstance(result, dict)
    for key in ("estimate", "z", "v_theta", "lam", "plies", "trajectory"):
        assert key in result

    # Independent computation of the mixed leaf value from the literature formula:
    # estimate = (1 - lam) * v_theta + lam * z
    vt_indep = float(value_net(state))
    z_indep = float(outcome(state))
    lam_indep = float(lam)
    expected_estimate = (1.0 - lam_indep) * vt_indep + lam_indep * z_indep

    assert result["z"] == z_indep
    assert result["v_theta"] == vt_indep
    assert result["lam"] == lam_indep
    assert result["estimate"] == expected_estimate

    # Trajectory starts at the initial state and grows by one per ply
    assert result["trajectory"][0] == state
    assert len(result["trajectory"]) == result["plies"] + 1
    assert len(result["actions"]) == result["plies"]


def test_alfgom_edge():
    """Test that with no value_net, lambda is forced to 1 (pure rollout)."""
    state = 7
    horizon = 3

    def rollout_net(s):
        return [1.0, 0.0]

    def step(s, a):
        return s + 10

    def outcome(s):
        return -0.75

    result = alphago_montecarlo(
        state,
        rollout_net,
        horizon=horizon,
        step=step,
        outcome=outcome,
        # value_net omitted -> pure rollout
        lam=0.5,
    )

    assert isinstance(result, dict)

    # Pure rollout means estimate == z and v_theta is NaN
    z_indep = float(outcome(state))
    assert result["v_theta"] != result["v_theta"]  # NaN check
    assert result["z"] == z_indep
    assert result["estimate"] == z_indep
    assert result["plies"] == horizon
