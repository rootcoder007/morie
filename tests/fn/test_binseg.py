"""Anchored tests for binseg (Scott-Knott 1974; Killick et al. 2012)."""

from morie.fn.binseg import binseg, binary_segmentation

X1 = [0.1, -0.2, 0.05, 0.3, -0.1, 5.2, 4.9, 5.1, 5.3, 4.8,
      1.9, 2.1, 2.0, 1.8, 2.2]


def test_binseg_matches_changepoint_package_anchor():
    # Anchor: changepoint::cpt.mean(x1, method="BinSeg", Q=2,
    # penalty="Manual", pen.value=0) returns cpts {5, 10}
    # (run 2026-08-09, R 4.6.1).
    r = binseg(X1, 2)
    assert r["changepoints"] == [5, 10]
    assert r["order"] == [5, 10]  # big jump found first


def test_binseg_gain_arithmetic():
    # First split gain must equal C(y_1:n) - C(y_1:tau) - C(y_tau+1:n)
    # computed independently.
    r = binseg(X1, 1)
    tau = r["order"][0]

    def ssdev(seg):
        mu = sum(seg) / len(seg)
        return sum((v - mu) ** 2 for v in seg)

    gain = ssdev(X1) - ssdev(X1[:tau]) - ssdev(X1[tau:])
    assert abs(r["improvements"][0] - gain) < 1e-12


def test_binseg_first_split_is_best_single_split():
    # the first BS split must be the argmax of the single-split gain
    def ssdev(seg):
        mu = sum(seg) / len(seg)
        return sum((v - mu) ** 2 for v in seg)

    gains = {t: ssdev(X1) - ssdev(X1[:t]) - ssdev(X1[t:])
             for t in range(1, len(X1))}
    best = max(gains, key=lambda t: gains[t])
    r = binseg(X1, 1)
    assert r["order"][0] == best


def test_alias():
    assert binary_segmentation(X1, 2)["changepoints"] == \
        binseg(X1, 2)["changepoints"]
