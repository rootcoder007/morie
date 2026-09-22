"""Tests for alf3df.af3_diffusion_step."""

from morie.fn import _array_core as np

from morie.fn.alf3df import af3_diffusion_step


def test_alf3df_basic():
    """Test basic functionality with a constant denoiser."""
    x = np.random.default_rng(42).normal(0.0, 1.0, (4, 3))
    t = 5.0
    sigma_next = 1.0
    gamma = 0.0
    noise_scale = 1.0
    step_scale = 1.0

    # Constant denoiser: returns the clean coordinates as-is.
    score_fn = lambda v, s: [[0.0, 0.0, 0.0] for _ in range(len(v))]

    result = af3_diffusion_step(
        x, t, score_fn,
        sigma_next=sigma_next,
        gamma=gamma,
        noise_scale=noise_scale,
        step_scale=step_scale,
        order="euler",
        rng=None,
        augment=False,
    )

    assert isinstance(result, dict)
    # The function returns the next coordinates, the churned level, and the direction.
    assert "x" in result
    assert "sigma_hat" in result
    assert "direction" in result

    out = result["x"]

    # With a constant (zero) denoiser, the EDM direction is (cur - den) / that = cur / that.
    # Then nxt = cur + dt * (cur / that), with that = t * (1 + gamma) = t and dt = sigma_next - t.
    that = t * (1.0 + gamma)
    dt = (sigma_next - that) * step_scale

    cur = [[float(v) for v in row] for row in x]
    expected = [[cur[i][c] + dt * (cur[i][c] / that) for c in range(3)]
                for i in range(len(cur))]

    assert len(out) == len(expected)
    for r_out, r_exp in zip(out, expected):
        assert len(r_out) == 3
        for a, b in zip(r_out, r_exp):
            assert abs(a - b) < 1e-9

    assert abs(result["sigma_hat"] - that) < 1e-12

    # The returned direction at the start of the step, d = (cur - 0) / that = cur / that.
    expected_d = [[cur[i][c] / that for c in range(3)] for i in range(len(cur))]
    for r_out, r_exp in zip(result["direction"], expected_d):
        for a, b in zip(r_out, r_exp):
            assert abs(a - b) < 1e-12


def test_alf3df_edge():
    """Test edge cases: final step with sigma_next omitted, and gamma=0 is reproducible."""
    x = np.random.default_rng(42).normal(0.0, 1.0, (4, 3))
    t = 5.0
    score_fn = lambda v, s: [[0.0, 0.0, 0.0] for _ in range(len(v))]

    # Final step: sigma_next omitted defaults to 0.0, order 'euler' skips Heun correction.
    result = af3_diffusion_step(
        x, t, score_fn,
        sigma_next=None,
        gamma=0.0,
        order="euler",
        rng=None,
        augment=False,
    )

    assert isinstance(result, dict)
    assert "x" in result
    assert "sigma_hat" in result
    assert "direction" in result

    # With denoiser identically zero, nxt = cur + dt * (cur / t), dt = 0 - t = -t, so nxt = 0.
    cur = [[float(v) for v in row] for row in x]
    for row in result["x"]:
        assert len(row) == 3
        for v in row:
            assert abs(v) < 1e-9
    assert abs(result["sigma_hat"] - t) < 1e-12
