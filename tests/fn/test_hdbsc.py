"""Tests for hdbsc (HDBSCAN*, Campello, Moulavi & Sander 2013).

Replaces the generated stub, which imported ``hdbscan``.
"""

from morie.fn.hdbsc import hdbsc


def _blobs(per=20, sep=10.0):
    pts = []
    for cx, cy in ((0.0, 0.0), (sep, 0.0), (0.0, sep)):
        for i in range(per):
            pts.append([cx + (i % 5) * 0.2, cy + (i // 5) * 0.2])
    return pts


def test_three_well_separated_blobs_are_found():
    res = hdbsc(_blobs(), min_pts=4, min_cluster_size=5)
    assert res["n_clusters"] == 3
    labels = res["labels"]
    # each blob keeps one label
    for start in (0, 20, 40):
        block = labels[start:start + 20]
        assert len(set(block)) == 1
        assert block[0] != -1


def test_noise_is_labelled_minus_one():
    pts = _blobs() + [[100.0, 100.0], [-80.0, 55.0]]
    res = hdbsc(pts, min_pts=4, min_cluster_size=5)
    assert res["labels"][-1] == -1 and res["labels"][-2] == -1


def test_core_distances_are_reported_per_point():
    pts = _blobs()
    res = hdbsc(pts, min_pts=4, min_cluster_size=5)
    assert len(res["core_distances"]) == len(pts)
    assert all(d >= 0 for d in res["core_distances"])


def test_a_larger_min_cluster_size_merges_or_drops_small_groups():
    pts = _blobs(per=8)
    few = hdbsc(pts, min_pts=3, min_cluster_size=3)["n_clusters"]
    many = hdbsc(pts, min_pts=3, min_cluster_size=20)["n_clusters"]
    assert many <= few


def test_both_selection_rules_run():
    pts = _blobs()
    for sel in ("eom", "leaf"):
        res = hdbsc(pts, min_pts=4, min_cluster_size=5, selection=sel)
        assert res["selection"] == sel
        assert res["n_clusters"] >= 1


def test_stabilities_are_non_negative():
    res = hdbsc(_blobs(), min_pts=4, min_cluster_size=5)
    assert all(v >= 0 for v in res["stabilities"].values())


def test_validation():
    for call in (lambda: hdbsc(_blobs(), selection="dbscan"),
                 lambda: hdbsc([[0.0, 0.0]], min_pts=5)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
