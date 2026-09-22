"""Tests for agalfsy.alphazero_alphafold_synergy."""

import math

from morie.fn import _array_core as np

from morie.fn.agalfsy import alphazero_alphafold_synergy


def test_agalfsy_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # receptor: shape (m, 3) -- m protein atoms, each with x, y, z
    protein = rng.normal(0, 1, (40, 3))
    # ligand: shape (n, 3) -- n ligand atoms, each with x, y, z
    ligand = rng.normal(0, 1, (8, 3))
    # site: shape (n, 3) -- reference pose matching ligand atom count
    site = ligand + rng.normal(0, 0.1, (8, 3))

    result = alphazero_alphafold_synergy(
        protein, ligand, site=site, max_steps=20, min_steps=10, window=5
    )
    assert isinstance(result, dict)
    # the documented keys (all of these are produced by rl_pose_search)
    assert "pose" in result
    assert "rmsd" in result
    assert "dcc" in result
    assert "success" in result
    assert "improved" in result
    assert "steps" in result
    assert "stop_reason" in result
    assert "reward_total" in result
    assert "trajectory" in result
    assert "policy_kind" in result
    # with the built-in greedy policy, policy_kind must name it as an oracle
    assert "oracle" in result["policy_kind"]
    # n_actions is fixed by the paper at 12
    assert result["n_actions"] == 12
    # steps must be a non-negative integer bounded by max_steps
    assert isinstance(result["steps"], int)
    assert 0 <= result["steps"] <= 20
    # reward_total is a non-negative float: reward = exp(-d/18) - exp(-d_prev/18)
    # with negatives doubled, so sum over the run is >= 0
    assert result["reward_total"] >= 0.0


def test_agalfsy_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    protein = rng.normal(0, 1, (40, 3))
    ligand = rng.normal(0, 1, (8, 3))
    site = ligand + rng.normal(0, 0.1, (8, 3))

    result = alphazero_alphafold_synergy(
        protein, ligand, site=site, max_steps=20, min_steps=10, window=5
    )
    assert isinstance(result, dict)
    # independently recompute the success criterion from the returned pose and
    # site: DCC is the distance between ligand centroid and site centroid, and
    # the paper flags success when DCC < 4 A.
    pose = result["pose"]
    pose_c = [sum(p[i] for p in pose) / len(pose) for i in range(3)]
    site_c = [sum(s[i] for s in site) / len(site) for i in range(3)]
    expected_dcc = math.sqrt(sum((pose_c[i] - site_c[i]) ** 2 for i in range(3)))
    expected_success = expected_dcc < 4.0
    assert result["dcc"] == expected_dcc
    assert result["success"] is expected_success
    # box=18 A default: centroid displacement must never exceed half the box
    # at any returned step (the search would have terminated with left_box
    # otherwise).
    start_c = [sum(p[i] for p in ligand) / len(ligand) for i in range(3)]
    assert max(abs(pose_c[i] - start_c[i]) for i in range(3)) <= 18.0 / 2.0 + 1e-9
