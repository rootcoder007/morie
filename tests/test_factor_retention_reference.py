"""Factor retention and ML factor analysis: factanal, Velicer/O'Connor MAP, nFactors::nScree, Philox parallel analysis."""

import math

from morie.fn.efa2 import efa_nfactors
from morie.fn.mlfac import mlfac


def two_factor_data():
    xs = 99

    def u():
        nonlocal xs
        xs = xs * 16807 % 2147483647
        return xs / 2147483647

    def z():
        return math.sqrt(-2 * math.log(u())) * math.cos(2 * math.pi * u())

    X = []
    for _ in range(300):
        f1, f2 = z(), z()
        X.append(
            [
                0.8 * f1 + 0.4 * z(),
                0.7 * f1 + 0.5 * z(),
                0.6 * f1 + 0.6 * z(),
                0.8 * f2 + 0.4 * z(),
                0.7 * f2 + 0.5 * z(),
                0.5 * f1 + 0.5 * f2 + 0.5 * z(),
            ]
        )
    return X


def test_ml_factor_analysis_matches_factanal():
    X = two_factor_data()
    r = mlfac(X, n_factors=2)
    assert (
        max(
            abs(a - b)
            for a, b in zip(
                r["uniqueness"],
                [
                    0.246269453065463,
                    0.260086455057108,
                    0.525380287526848,
                    0.206369465646447,
                    0.408501546403968,
                    0.343808068939422,
                ],
            )
        )
        < 1e-8
    )  # factanal(X, 2)$uniquenesses
    assert abs(r["statistic"] - 3.37122551114991) < 1e-8 and r["dof"] == 4  # factanal STATISTIC, dof
    assert abs(sum(v * v for v in r["loadings"][0]) + r["uniqueness"][0] - 1) < 1e-10  # standardised: h2 + psi = 1


def test_retention_criteria_match_references():
    X = two_factor_data()
    m = efa_nfactors(X, method="map")
    assert (
        max(
            abs(a - b)
            for a, b in zip(
                m["map_values"],
                [0.191906509332262, 0.176833764100459, 0.105901901182621, 0.233252358695946, 0.498113956813281],
            )
        )
        < 1e-12
    )  # O'Connor (2000) MAP
    m4 = efa_nfactors(X, method="map4")
    assert (
        max(
            abs(a - b)
            for a, b in zip(
                m4["map_values"],
                [0.0706261926535189, 0.0450981522212789, 0.0270119877599666, 0.109960722714157, 0.397868835874134],
            )
        )
        < 1e-12
    )
    b = efa_nfactors(X, method="bic")
    assert (
        max(abs(a - c) for a, c in zip(b["bic_values"][:2], [263.279132684515, -19.4439043874749])) < 1e-8
    )  # factanal chi2 - df log n
    a = efa_nfactors(X, method="aic")
    assert max(abs(v - c) for v, c in zip(a["aic_values"][:2], [296.61317495642, -4.62877448885009])) < 1e-8
    p = efa_nfactors(X, method="parallel", nsim=20, seed=7)
    assert (
        max(
            abs(v - c)
            for v, c in zip(
                p["threshold"],
                [
                    1.2565893175643983,
                    1.1449133150508983,
                    1.085070600812775,
                    0.995374529325129,
                    0.9656585304195988,
                    0.8628327543397473,
                ],
            )
        )
        < 1e-12
    )  # Philox, identical in the R arm
    for meth in ("parallel", "map", "map4", "kaiser", "scree", "af", "variance", "aic", "bic"):
        assert (
            efa_nfactors(X, method=meth)["n_factors"] == 2
        ), meth  # the data have two factors (nScree: noc = naf = nkaiser = 2)
    s = efa_nfactors(X, method="scree")
    ev = s["eigenvalues"]
    assert abs(s["predicted"][0] - (ev[1] - (ev[5] - ev[1]) / 4)) < 1e-15  # line through (2, l2) and (6, l6) at 1


def test_parallel_analysis_rejects_noise():
    xs = 2024

    def u():
        nonlocal xs
        xs = xs * 16807 % 2147483647
        return xs / 2147483647

    N = [[math.sqrt(-2 * math.log(u())) * math.cos(2 * math.pi * u()) for _ in range(6)] for _ in range(120)]
    pa = efa_nfactors(N, method="parallel", nsim=30, seed=1)
    lead = next((j for j, (e, t) in enumerate(zip(pa["eigenvalues"], pa["threshold"])) if e <= t), 6)
    assert pa["n_factors"] == lead == 0  # no observed eigenvalue beats its simulated 95th percentile
    assert efa_nfactors(N, method="kaiser")["n_factors"] == 3  # while three sampling-noise eigenvalues exceed 1
