"""Tests for glides.glide_score_proxy."""

import math

from morie.fn import _array_core as np

from morie.fn.glides import glide_score_proxy


def test_glides_basic():
    """Test basic functionality with atom-row shaped inputs."""
    # receptor and ligand_pose must be sequences of atom rows: [x, y, z, type]
    receptor = [
        [0.0, 0.0, 0.0, "C"],
        [3.0, 0.0, 0.0, "O"],
        [0.0, 3.0, 0.0, "N"],
    ]
    ligand_pose = [
        [1.0, 0.0, 0.0, "C"],
        [2.0, 0.0, 0.0, "O"],
        [1.0, 2.0, 0.0, "N"],
    ]
    radii = [("C", 1.7), ("O", 1.52), ("N", 1.55)]
    depths = [("C", 0.1), ("O", 0.2), ("N", 0.15)]
    charges = [("C", 0.0), ("O", -0.4), ("N", -0.2)]
    lipophilic = ["C"]
    hbonds = []

    result = glide_score_proxy(
        receptor, ligand_pose,
        radii=radii, depths=depths, charges=charges,
        lipophilic=lipophilic, hbonds=hbonds,
    )

    assert isinstance(result, dict)
    assert "gscore" in result
    assert "estimate" in result
    assert result["gscore"] == result["estimate"]
    # n_contacts == len(receptor) * len(ligand_pose) when no cutoff
    assert result["n_contacts"] == 3 * 3
    assert result["n_lipophilic"] >= 0
    assert result["dielectric"] == "constant"
    assert result["method"] == "Glide-style empirical docking score"

    # Independent numerical sanity check on the rotatable-bond penalty.
    e_rot = 0.35 * 0
    result_rot = glide_score_proxy(
        receptor, ligand_pose,
        radii=radii, depths=depths, charges=charges,
        lipophilic=lipophilic, hbonds=hbonds,
        n_rot=0, rot_penalty=0.35,
    )
    assert result_rot["rot_penalty"] == e_rot

    # With rot_penalty and n_rot both > 0 the documented contribution is
    # rot_penalty * n_rot, raised positively.
    e_rot_nonzero = 0.35 * 2
    result_flex = glide_score_proxy(
        receptor, ligand_pose,
        radii=radii, depths=depths, charges=charges,
        lipophilic=lipophilic, hbonds=hbonds,
        n_rot=2, rot_penalty=0.35,
    )
    assert result_flex["rot_penalty"] == e_rot_nonzero


def test_glides_edge():
    """Test edge cases (empty ligand)."""
    receptor = [
        [0.0, 0.0, 0.0, "C"],
        [3.0, 0.0, 0.0, "O"],
    ]
    ligand_pose = []  # no ligand atoms -> no contacts
    radii = [("C", 1.7), ("O", 1.52)]
    depths = [("C", 0.1), ("O", 0.2)]
    charges = [("C", 0.0), ("O", -0.4)]

    result = glide_score_proxy(
        receptor, ligand_pose,
        radii=radii, depths=depths, charges=charges,
    )
    assert isinstance(result, dict)
    assert result["n_contacts"] == 0
    assert result["n_lipophilic"] == 0
    assert math.isnan(result["se"])
