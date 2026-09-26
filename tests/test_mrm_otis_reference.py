"""OTIS / Mandela classifiers against the R arm."""

from morie import mrm_otis as O
from morie.fn import _frame_core as pd


def _b01():
    rows = []
    for i in range(400):
        rows.append(
            {
                "UniqueIndividual_ID": f"P{i % 150}",
                "EndFiscalYear": 2022 + i % 3,
                "NumberConsecutiveDays_Segregation": (i * 37) % 40 + 1,
                "MentalHealth_Alert": "Yes" if (i * 7) % 5 < 2 else "No",
                "SuicideRisk_Alert": "Yes" if (i * 11) % 7 < 2 else "No",
                "SuicideWatch_Alert": "Yes" if (i * 13) % 9 < 2 else "No",
            }
        )
    return pd.DataFrame(rows)


def test_classify_mandela_matches_r_arm():
    d = _b01()
    # R arm, mrm_classify_mandela(broader_rc = TRUE)
    ref = {
        "row": ([83, 83, 84, 250], [98, 98, 87, 283], [134, 133, 133, 400]),
        "individual_any": ([48, 47, 48, 143], [49, 49, 48, 146], [50, 50, 50, 150]),
        "individual_cumulative": ([50, 48, 49, 147], [50, 48, 49, 147], [50, 50, 50, 150]),
    }
    for dn, (m, b, den) in ref.items():
        r = O.mrm_classify_mandela(d, denominator=dn, broader_rc=True)
        assert list(r["n_mandela"]) == m
        assert list(r["n_broader_rc"]) == b
        assert list(r["denominator"]) == den


def test_broader_counts_alerts_regardless_of_duration():
    d = pd.DataFrame(
        {
            "UniqueIndividual_ID": ["a", "b", "c", "d"],
            "EndFiscalYear": [2024] * 4,
            "NumberConsecutiveDays_Segregation": [20, 3, 3, 3],
            "MentalHealth_Alert": ["No", "Yes", "Yes", "No"],
            "SuicideRisk_Alert": ["No", "Yes", "No", "No"],
            "SuicideWatch_Alert": ["No", "No", "No", "No"],
        }
    )
    r = O.mrm_classify_mandela(d, denominator="row", broader_rc=True)
    assert (list(r["n_mandela"])[0], list(r["n_broader_rc"])[0]) == (1, 2)


def test_seg_duration_survival_columns_match_r_arm():
    s = O.mrm_otis_seg_duration_km(_b01())
    got = [list(s[c])[0] for c in ("days_at_S50", "days_at_S25", "days_at_S10", "days_at_S05", "days_at_S01")]
    for a, b in zip(got, [20.5, 30.25, 36.1, 38.05, 40.0]):
        assert abs(a - b) < 1e-9
