"""Tests for causscd (Arkhangelsky et al. 2021, synthetic DID)."""

from morie.fn.causscd import causscd, sdid, time_weights, unit_weights

N, T, TPRE, TAU = 6, 8, 5, 3.0
ALPHA = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
TREATED = [False] * 5 + [True]


def _exact():
    return [[ALPHA[i] + 0.1 * t +
             (TAU if (TREATED[i] and t >= TPRE) else 0.0)
             for t in range(T)] for i in range(N)]


def _mismatched():
    Y = []
    for i in range(N):
        share = 1.0 if (i in (0, 1) or TREATED[i]) else 0.0
        Y.append([ALPHA[i] + 2.0 * t * share + 0.05 * t +
                  (TAU if (TREATED[i] and t >= TPRE) else 0.0)
                  for t in range(T)])
    return Y


def test_every_weighting_recovers_tau_on_an_exact_model():
    for m in ("did", "sc", "sdid"):
        assert abs(sdid(_exact(), TREATED, TPRE, m)["tau"] - TAU) < 1e-6


def test_the_weights_are_simplex_weights():
    w, _, _ = unit_weights(_exact(), TREATED, TPRE)
    assert abs(sum(w) - 1.0) < 1e-9
    assert all(v >= -1e-12 for v in w)
    assert abs(w[5]) < 1e-12                 # nothing on the treated unit
    lam, _ = time_weights(_exact(), TREATED, TPRE)
    assert abs(sum(lam) - 1.0) < 1e-9
    assert all(abs(lam[t]) < 1e-12 for t in range(TPRE, T))


def test_did_uses_the_flat_weights():
    res = sdid(_exact(), TREATED, TPRE, "did")
    assert all(abs(res["unit_weights"][i] - 0.2) < 1e-12 for i in range(5))
    assert all(abs(res["time_weights"][t] - 1.0 / TPRE) < 1e-12
               for t in range(TPRE))


def test_weighting_rescues_an_estimate_did_gets_wrong():
    r = causscd(_mismatched(), TREATED, TPRE)
    assert abs(r["did"] - TAU) > 4.0
    assert abs(r["sdid"] - TAU) < 0.6
    assert abs(r["sdid"] - TAU) < abs(r["sc"] - TAU)
    assert r["unit_weights"][0] > 0.3 and r["unit_weights"][1] > 0.3


def test_invariances():
    base = sdid(_exact(), TREATED, TPRE)["tau"]
    shifted = [[v + 100.0 for v in row] for row in _exact()]
    assert abs(sdid(shifted, TREATED, TPRE)["tau"] - base) < 1e-6
    scaled = [[2.0 * v for v in row] for row in _exact()]
    assert abs(sdid(scaled, TREATED, TPRE)["tau"] - 2.0 * base) < 1e-5
    shifted_units = [[row[t] + 7.0 * i for t in range(T)]
                     for i, row in enumerate(_exact())]
    assert abs(sdid(shifted_units, TREATED, TPRE)["tau"] - TAU) < 1e-5


def test_validation():
    for call in (lambda: sdid(_exact(), TREATED, 0),
                 lambda: sdid(_exact(), TREATED, T),
                 lambda: sdid(_exact(), [False] * N, TPRE),
                 lambda: sdid(_exact(), [True] * N, TPRE),
                 lambda: sdid(_exact(), TREATED[:-1], TPRE),
                 lambda: sdid(_exact(), TREATED, TPRE, method="bacon"),
                 lambda: sdid([[1.0, 2.0], [3.0]], [False, True], 1)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
