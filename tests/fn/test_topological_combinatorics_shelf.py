"""Topological combinatorics: homology mod 2 against known surfaces.

The complexes are not recalled from memory: the torus is generated as
the standard 7-vertex quotient {i, i+1, i+3} + {i, i+2, i+3} mod 7,
and the 6-vertex projective plane was FOUND by backtracking search
over the triangles of K6 for a subcomplex with every edge in exactly
two triangles -- then certified by the homology computation itself
(b = 1, 1, 1 over F_2 with chi = 1 pins RP^2 uniquely among closed
surfaces). Sperner is asserted ODD, not merely nonzero, over 200
random admissible labellings driven by the shared LCG.

Sources: Sperner (1928); Munkres (1984) *Elements of Algebraic
Topology*; Matousek (2003) *Using the Borsuk-Ulam Theorem*.
"""

import pytest

from morie.fn.topcmb import (
    betti_numbers_gf2,
    euler_characteristic,
    simplicial_complex_faces,
    sperner_lemma_triangle,
)

TORUS = [tuple(sorted((i % 7, (i + 1) % 7, (i + 3) % 7))) for i in range(7)] \
    + [tuple(sorted((i % 7, (i + 2) % 7, (i + 3) % 7))) for i in range(7)]

RP2 = [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
       (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5)]

SPHERE = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

CIRCLE = [(0, 1), (1, 2), (0, 2)]


# --------------------------------------------------------------------
# Face closure and Euler characteristic
# --------------------------------------------------------------------

def test_face_closure_of_a_triangle():
    faces = simplicial_complex_faces([(0, 1, 2)])
    assert faces[0] == [(0,), (1,), (2,)]
    assert faces[1] == [(0, 1), (0, 2), (1, 2)]
    assert faces[2] == [(0, 1, 2)]


def test_euler_characteristic_of_the_classics():
    assert euler_characteristic([(0, 1, 2, 3)])["chi"] == 1     # ball
    assert euler_characteristic(SPHERE)["chi"] == 2
    assert euler_characteristic(CIRCLE)["chi"] == 0
    assert euler_characteristic(TORUS)["chi"] == 0
    assert euler_characteristic(RP2)["chi"] == 1


def test_the_torus_f_vector_is_the_minimal_one():
    out = euler_characteristic(TORUS)
    assert out["f_vector"] == [7, 21, 14]   # Csaszar: every pair an edge


def test_face_validation():
    with pytest.raises(ValueError, match="no simplices"):
        simplicial_complex_faces([])
    with pytest.raises(ValueError, match="empty simplices"):
        simplicial_complex_faces([()])


# --------------------------------------------------------------------
# Betti numbers over F_2
# --------------------------------------------------------------------

def test_betti_numbers_of_the_classical_surfaces():
    assert betti_numbers_gf2(CIRCLE)["betti"] == [1, 1]
    assert betti_numbers_gf2(SPHERE)["betti"] == [1, 0, 1]
    assert betti_numbers_gf2(TORUS)["betti"] == [1, 2, 1]
    # over F_2 the projective plane has FULL homology 1, 1, 1 -- the
    # Z-torsion in H_1 becomes a genuine F_2 class, and b_2 = 1 even
    # though RP^2 is non-orientable, because orientability is
    # invisible mod 2
    assert betti_numbers_gf2(RP2)["betti"] == [1, 1, 1]


def test_a_contractible_complex_has_trivial_homology():
    out = betti_numbers_gf2([(0, 1, 2, 3)])
    assert out["betti"] == [1, 0, 0, 0]


def test_disconnected_components_add_in_b0():
    assert betti_numbers_gf2([(0, 1), (2, 3)])["betti"][0] == 2
    assert betti_numbers_gf2(CIRCLE + [(3, 4), (4, 5), (3, 5)])["betti"] \
        == [2, 2]


def test_a_wedge_of_two_circles_has_b1_of_2():
    wedge = [(0, 1), (1, 2), (0, 2), (0, 3), (3, 4), (0, 4)]
    assert betti_numbers_gf2(wedge)["betti"] == [1, 2]


def test_the_boundary_map_squares_to_zero_everywhere():
    for cx in (SPHERE, TORUS, RP2, [(0, 1, 2, 3)], [(0, 1, 2, 3, 4)]):
        out = betti_numbers_gf2(cx)
        assert out["boundary_squared_zero"] is True
        assert not out.warnings


def test_the_euler_poincare_identity_holds_on_every_fixture():
    for cx in (CIRCLE, SPHERE, TORUS, RP2, [(0, 1, 2, 3)],
               [(0, 1), (2, 3)]):
        out = betti_numbers_gf2(cx)
        assert out["euler_poincare_holds"] is True
        assert out["chi_from_faces"] == out["chi_from_betti"]


def test_filling_the_circle_kills_b1():
    assert betti_numbers_gf2(CIRCLE)["betti"] == [1, 1]
    assert betti_numbers_gf2([(0, 1, 2)])["betti"] == [1, 0, 0]


def test_the_rp2_fixture_really_is_a_closed_surface():
    # every edge in exactly two triangles -- the property the search
    # selected for, re-verified here rather than trusted
    from collections import Counter
    from itertools import combinations
    edge_count = Counter()
    for t in RP2:
        for e in combinations(t, 2):
            edge_count[e] += 1
    assert all(v == 2 for v in edge_count.values())
    assert len(edge_count) == 15


# --------------------------------------------------------------------
# Sperner's lemma
# --------------------------------------------------------------------

def test_the_rainbow_count_is_odd_at_every_subdivision():
    for k in range(1, 13):
        out = sperner_lemma_triangle(k)
        assert out["is_odd"] is True
        assert out["n_cells"] == k * k


def test_oddness_survives_200_random_admissible_labellings():
    # labels drawn by the shared LCG so the R parity suite sees the
    # same labellings bit for bit
    def lcg_labels(k, seed):
        s = seed
        lab = {}
        for i in range(k + 1):
            for j in range(k + 1 - i):
                allowed = []
                if k - i - j > 0:
                    allowed.append(0)
                if i > 0:
                    allowed.append(1)
                if j > 0:
                    allowed.append(2)
                s = (1664525 * s + 1013904223) % 2 ** 32
                lab[(i, j)] = allowed[s % len(allowed)]
        return lab
    for seed in range(200):
        k = 2 + seed % 7
        out = sperner_lemma_triangle(k, lcg_labels(k, seed))
        assert out["is_odd"] is True


def test_an_improper_labelling_is_refused_not_computed():
    # corner (k, 0) must carry label 1; forcing 0 there breaks the
    # Sperner condition, and the lemma is FALSE for improper labellings
    k = 2
    labels = {}
    for i in range(k + 1):
        for j in range(k + 1 - i):
            allowed = ([0] if k - i - j > 0 else []) + \
                ([1] if i > 0 else []) + ([2] if j > 0 else [])
            labels[(i, j)] = allowed[0]
    labels[(k, 0)] = 0
    with pytest.raises(ValueError, match="not admissible"):
        sperner_lemma_triangle(k, labels)


def test_a_missing_label_is_an_error():
    with pytest.raises(ValueError, match="no label supplied"):
        sperner_lemma_triangle(2, {(0, 0): 0})


def test_sperner_validation():
    with pytest.raises(ValueError, match="must be positive"):
        sperner_lemma_triangle(0)
