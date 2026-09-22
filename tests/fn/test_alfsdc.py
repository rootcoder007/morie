"""Tests for alfsdc.alphafold_sidechain."""

from morie.fn import _array_core as np

from morie.fn.alfsdc import alphafold_sidechain


def _make_R(angle_rad):
    """Build a proper 3x3 rotation about z-axis by angle_rad."""
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def test_alfsdc_basic():
    """Test basic functionality with a single residue, single torsion frame."""
    # Seed RNG for reproducibility.
    rng = np.random.default_rng(42)

    n = 1  # one residue
    nf = 1  # one torsion frame

    # One backbone frame: random proper rotation R plus a translation t.
    Rb = _make_R(0.1)
    tb = [float(rng.normal()), float(rng.normal()), float(rng.normal())]
    frames = [[Rb, tb]]

    # Torsion angles: n x nf x 2 (cos, sin), unnormalised.
    cos_a = float(rng.normal())
    sin_a = float(rng.normal())
    angles = [[[cos_a, sin_a]]]

    # One torsion frame whose parent is the backbone frame (parent = -1).
    Rl = _make_R(0.3)
    tl = [0.0, 0.0, 0.0]
    littf = [[Rl, tl]]

    parent = [-1]

    # One atom, idealised in its own frame.
    litx = [[1.5, 0.0, 0.0]]
    frameof = [0]

    result = alphafold_sidechain(frames, angles, littf, parent, litx, frameof)

    # The result is a RichResult; allow dict-like access.
    # It must expose 'x', 'frames', 'estimate', 'n', 'method'.
    assert "x" in result
    assert "frames" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    # n is the number of residues passed in.
    assert int(result["n"]) == n

    # Compute the expected atom coordinate from the formula written out.
    # Composed torsion frame:
    #   tf = Rb Rl  then  * rotx(theta),  translation  Rb (Rl x_t + tl) + tb
    # Atom: tf applied to litx = [1.5, 0, 0].
    import math
    theta = math.atan2(sin_a, cos_a)
    cx, sx = math.cos(theta), math.sin(theta)

    # rotx(theta) acting on [1.5, 0, 0] -> [1.5, 0, 0].
    rx_litx = [1.5, 0.0, 0.0]

    # Apply Rl to rx_litx, then add tl, then apply Rb, then add tb.
    Rl_x = [
        Rl[0][0] * rx_litx[0] + Rl[0][1] * rx_litx[1] + Rl[0][2] * rx_litx[2],
        Rl[1][0] * rx_litx[0] + Rl[1][1] * rx_litx[1] + Rl[1][2] * rx_litx[2],
        Rl[2][0] * rx_litx[0] + Rl[2][1] * rx_litx[1] + Rl[2][2] * rx_litx[2],
    ]
    after_l = [Rl_x[0] + tl[0], Rl_x[1] + tl[1], Rl_x[2] + tl[2]]
    Rb_y = [
        Rb[0][0] * after_l[0] + Rb[0][1] * after_l[1] + Rb[0][2] * after_l[2],
        Rb[1][0] * after_l[0] + Rb[1][1] * after_l[1] + Rb[1][2] * after_l[2],
        Rb[2][0] * after_l[0] + Rb[2][1] * after_l[1] + Rb[2][2] * after_l[2],
    ]
    expected = [Rb_y[0] + tb[0], Rb_y[1] + tb[1], Rb_y[2] + tb[2]]

    got = result["x"][0][0]
    assert abs(got[0] - expected[0]) < 1e-9
    assert abs(got[1] - expected[1]) < 1e-9
    assert abs(got[2] - expected[2]) < 1e-9

    # Composed frame is a proper rotation: columns orthonormal, det = +1.
    tf_R = result["frames"][0][0][0]
    # dot of col 0 with itself
    d00 = tf_R[0][0] ** 2 + tf_R[1][0] ** 2 + tf_R[2][0] ** 2
    assert abs(d00 - 1.0) < 1e-9


def test_alfsdc_edge():
    """Test edge cases: zero torsion angle and identity literature transform."""
    n = 2
    nf = 1
    Rb = _make_R(0.2)
    tb = [1.0, 2.0, 3.0]
    frames = [[Rb, tb], [Rb, [tb[0] + 1.0, tb[1], tb[2]]]]

    # Zero torsion: (cos, sin) = (1, 0).
    angles = [[[1.0, 0.0]], [[1.0, 0.0]]]

    # Identity literature transform.
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    littf = [[I3, [0.0, 0.0, 0.0]]]
    parent = [-1]
    litx = [[2.0, 0.0, 0.0]]
    frameof = [0]

    result = alphafold_sidechain(frames, angles, littf, parent, litx, frameof)

    # With zero torsion and identity littf, atom = Rb * litx + tb.
    expected0 = [
        Rb[0][0] * 2.0 + tb[0],
        Rb[1][0] * 2.0 + tb[1],
        Rb[2][0] * 2.0 + tb[2],
    ]
    expected1 = [
        Rb[0][0] * 2.0 + tb[0] + 1.0,
        Rb[1][0] * 2.0 + tb[1],
        Rb[2][0] * 2.0 + tb[2],
    ]
    g0 = result["x"][0][0]
    g1 = result["x"][1][0]
    assert abs(g0[0] - expected0[0]) < 1e-9
    assert abs(g0[1] - expected0[1]) < 1e-9
    assert abs(g0[2] - expected0[2]) < 1e-9
    assert abs(g1[0] - expected1[0]) < 1e-9
    assert abs(g1[1] - expected1[1]) < 1e-9
    assert abs(g1[2] - expected1[2]) < 1e-9
