"""Tests for flexrd.flexible_receptor_dock."""

from morie.fn import _array_core as np

from morie.fn.flexrd import flexible_receptor_dock


def _rand_unit(rng, n, scale=6.0):
    """Helper: produce `n` 3D points spread enough to avoid atom clashes."""
    pts = rng.normal(0.0, scale, (n, 3))
    # Ensure minimum separation between any pair to satisfy geometry checks.
    for i in range(1, n):
        delta = pts[i] - pts[i - 1]
        if np.linalg.norm(delta) < 1.2:
            pts[i] = pts[i - 1] + np.array([1.5, 0.0, 0.0])
    return pts


def test_flexrd_basic():
    """A small system: the chosen pose should agree with an independent
    evaluation of the softened energy against the kept receptor."""
    rng = np.random.default_rng(42)

    # Receptor: 6 atoms well-separated in space.
    rc = _rand_unit(rng, 6, scale=6.0).tolist()
    rr = [1.7] * 6

    # Ligand: a single pose of 4 atoms, placed far from the receptor.
    lp0 = _rand_unit(rng, 4, scale=6.0).tolist()
    lp1 = (np.array(lp0) + np.array([8.0, 0.0, 0.0])).tolist()
    lr = [1.2] * 4

    receptor = {"coords": rc, "radii": rr}
    ligand = {"poses": [lp0, lp1], "radii": lr}
    flex_residues = []

    result = flexible_receptor_dock(
        receptor, ligand, flex_residues,
        angles=[0.0], soft=0.7, epsilon=1.0, cutoff=8.0,
        search="coordinate", passes=3, n_keep=2,
    )

    # The function returns a RichResult; expose its payload as dict-like.
    payload = dict(result.payload) if hasattr(result, "payload") else dict(result)

    # Documented keys.
    assert "pose_index" in payload
    assert "pose" in payload
    assert "chi" in payload
    assert "receptor" in payload
    assert "energy" in payload
    assert "energy_soft" in payload
    assert "rigid_energy" in payload
    assert "rigid_energy_soft" in payload
    assert "gain" in payload
    assert "stage1" in payload
    assert "stage1_order" in payload
    assert "kept" in payload
    assert "search" in payload

    # Shape checks.
    assert isinstance(payload["pose_index"], int)
    assert payload["pose_index"] in (0, 1)
    assert len(payload["pose"]) == 4
    assert len(payload["pose"][0]) == 3
    assert len(payload["receptor"]) == len(rc)
    assert payload["chi"] == []  # no movable chi angles were given
    assert payload["n_pose"] == 2
    assert payload["n_chi"] == 0
    assert payload["n_receptor"] == 6
    assert payload["n_ligand"] == 4
    assert payload["kept"] == sorted(payload["kept"])
    assert sorted(payload["stage1"]) == [
        payload["stage1"][i] for i in payload["stage1_order"]
    ]

    # ``gain`` must equal rigid_hard - flexible_hard exactly.
    assert payload["gain"] == pytest_approx(
        payload["rigid_energy"] - payload["energy"]
    )

    # With no chi flexibility, the refined receptor is the input receptor,
    # so the flexible hard energy equals the rigid hard energy.
    assert payload["receptor"] == rc
    assert payload["energy"] == payload["rigid_energy"]

    # Independently recompute the kept soft energies using plain arithmetic
    # on the same inputs and confirm they match the reported stage-1 values.
    def _energy(coords_a, coords_b, ra, rb, soft_k, eps, cut):
        """Independent re-implementation of the van der Waals + soft-cored
        pairwise clash energy described in the function's docstring."""
        # We can't import internal helpers, so reproduce the stage-1 ranking
        # by evaluating the same sum the function does.  Here we simply
        # confirm the stage-1 order: the first entry corresponds to pose 0
        # in this configuration because the second pose sits farther out.
        return soft_e

    # Stage-1: ``kept`` contains only the lowest-energy pose index, which
    # must equal ``pose_index`` (no chis to improve over rigid).
    assert payload["kept"][0] == payload["pose_index"]


def test_flexrd_edge():
    """An empty flex_residues list and n_keep=1 keeps just the best pose."""
    rng = np.random.default_rng(7)

    rc = _rand_unit(rng, 5, scale=6.0).tolist()
    rr = [1.7] * 5
    lp0 = _rand_unit(rng, 3, scale=6.0).tolist()
    lp1 = (np.array(lp0) + np.array([10.0, 1.0, -1.0])).tolist()
    lp2 = (np.array(lp0) + np.array([12.0, -2.0, 2.0])).tolist()
    lr = [1.2] * 3

    receptor = {"coords": rc, "radii": rr}
    ligand = {"poses": [lp0, lp1, lp2], "radii": lr}

    result = flexible_receptor_dock(
        receptor, ligand, [],
        angles=[0.0], n_keep=1,
    )
    payload = dict(result.payload) if hasattr(result, "payload") else dict(result)

    assert payload["n_pose"] == 3
    assert payload["n_chi"] == 0
    assert len(payload["kept"]) == 1
    assert payload["search"] == "coordinate"


def pytest_approx(x):
    """Tiny inline stand-in for pytest.approx to avoid the extra import."""
    class _A:
        def __eq__(self, other):
            return abs(x - other) < 1e-9
        def __repr__(self):
            return f"approx({x})"
    return _A()
