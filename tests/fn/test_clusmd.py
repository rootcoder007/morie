"""Butina exclusion-sphere clustering."""
import importlib

import pytest

C = importlib.import_module("morie.fn.clusmd")
S = importlib.import_module("morie.fn.sasimi")


def block(base, drop):
    return set(range(base, base + 20)) - {base + d for d in drop}


GROUPS = [[block(b, d) for d in ([], [0], [1], [0, 1])]
          for b in (0, 100, 200, 300)]
FPS = [fp for g in GROUPS for fp in g]
TRUE = [i // 4 for i in range(16)]


def test_the_fixture_is_separated():
    within = min(S.tanimoto(a, b) for g in GROUPS for a in g
                 for b in g if a != b)
    across = max(S.tanimoto(a, b) for i, g in enumerate(GROUPS)
                 for j, h in enumerate(GROUPS) if i != j
                 for a in g for b in h)
    assert within > 0.8 > across


def test_butina_recovers_the_groups():
    r = C.butina_clustering(FPS, 0.8)
    assert r["n_clusters"] == 4
    got = r["assignment"]
    for i in range(16):
        for j in range(16):
            assert (got[i] == got[j]) == (TRUE[i] == TRUE[j])


@pytest.mark.parametrize("th", [0.5, 0.7, 0.8, 0.9, 0.95])
def test_every_member_is_within_the_threshold_of_its_centroid(th):
    for c in C.butina_clusters(FPS, th):
        for m in c["members"]:
            assert S.tanimoto(FPS[c["centroid"]], FPS[m]) >= th


def test_centroids_are_mutually_beyond_the_threshold():
    cents = [c["centroid"] for c in C.butina_clusters(FPS, 0.8)]
    for i, a in enumerate(cents):
        for b in cents[i + 1:]:
            assert S.tanimoto(FPS[a], FPS[b]) < 0.8


def test_the_clusters_partition_the_collection():
    members = [m for c in C.butina_clusters(FPS, 0.8)
               for m in c["members"]]
    assert sorted(members) == list(range(16))


def test_the_threshold_spans_both_extremes():
    assert C.butina_clustering(FPS, 0.0)["n_clusters"] == 1
    one = C.butina_clustering(FPS, 1.0)
    assert one["n_clusters"] == 16
    assert one["n_singletons"] == 16


def test_a_duplicate_clusters_even_at_one():
    dup = C.butina_clustering(FPS + [FPS[0]], 1.0)
    assert dup["n_clusters"] == 16
    assert max(dup["sizes"]) == 2


@pytest.mark.parametrize("recount", [False, True])
def test_both_recount_settings_keep_the_sphere_property(recount):
    skew = FPS + [block(0, [2]), block(0, [3]), block(0, [4])]
    for c in C.butina_clusters(skew, 0.8, recount):
        for m in c["members"]:
            assert S.tanimoto(skew[c["centroid"]], skew[m]) >= 0.8


def test_clustering_is_deterministic():
    assert C.butina_clustering(FPS, 0.8)["assignment"] \
        == C.butina_clustering(FPS, 0.8)["assignment"]


def test_the_summary_agrees_with_the_clusters():
    cl = C.butina_clusters(FPS, 0.8)
    s = C.cluster_summary(cl)
    assert s["n_compounds"] == 16
    assert sum(s["sizes"]) == 16
    assert s["centroids"] == [c["centroid"] for c in cl]


@pytest.mark.parametrize("call", [
    lambda: C.butina_clusters(FPS, 1.5),
    lambda: C.butina_clusters(FPS, -0.1),
    lambda: C.butina_clusters([], 0.8),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
