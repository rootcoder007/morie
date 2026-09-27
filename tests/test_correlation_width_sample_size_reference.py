"""Bonett-Wright sample sizes against Hedderich, Sachs & Reynarowych (2023) Table 7.85."""

from morie.fn.corwsn import correlation_width_sample_size


def test_table_7_85():
    table = [
        (0.10, 0.1, "pearson", 1507),
        (0.10, 0.2, "pearson", 378),
        (0.10, 0.2, "kendall", 168),
        (0.10, 0.3, "pearson", 168),
        (0.10, 0.3, "spearman", 169),
        (0.10, 0.3, "kendall", 77),
        (0.30, 0.1, "pearson", 1274),
        (0.30, 0.1, "spearman", 1331),
        (0.30, 0.1, "kendall", 560),
        (0.30, 0.2, "pearson", 320),
        (0.30, 0.2, "spearman", 334),
        (0.30, 0.2, "kendall", 143),
        (0.30, 0.3, "pearson", 143),
        (0.30, 0.3, "spearman", 149),
        (0.30, 0.3, "kendall", 65),
        (0.40, 0.1, "pearson", 1086),
        (0.40, 0.1, "spearman", 1173),
        (0.40, 0.2, "pearson", 273),
        (0.40, 0.2, "spearman", 295),
        (0.40, 0.2, "kendall", 122),
        (0.40, 0.3, "pearson", 123),
        (0.40, 0.3, "spearman", 132),
        (0.40, 0.3, "kendall", 57),
        (0.50, 0.1, "pearson", 867),
        (0.50, 0.1, "spearman", 975),
        (0.50, 0.1, "kendall", 382),
        (0.50, 0.2, "pearson", 219),
        (0.50, 0.2, "spearman", 246),
        (0.50, 0.2, "kendall", 99),
        (0.50, 0.3, "pearson", 99),
        (0.50, 0.3, "spearman", 111),
        (0.50, 0.3, "kendall", 46),
        (0.60, 0.1, "pearson", 633),
        (0.60, 0.1, "spearman", 746),
        (0.60, 0.1, "kendall", 280),
        (0.60, 0.2, "pearson", 161),
        (0.60, 0.2, "spearman", 189),
        (0.60, 0.2, "kendall", 73),
        (0.60, 0.3, "pearson", 74),
        (0.60, 0.3, "spearman", 86),
        (0.60, 0.3, "kendall", 35),
        (0.70, 0.1, "pearson", 404),
        (0.70, 0.1, "spearman", 503),
        (0.70, 0.1, "kendall", 180),
        (0.70, 0.2, "pearson", 105),
        (0.70, 0.2, "spearman", 129),
        (0.70, 0.2, "kendall", 49),
        (0.70, 0.3, "pearson", 49),
        (0.70, 0.3, "spearman", 60),
        (0.70, 0.3, "kendall", 24),
        (0.80, 0.1, "pearson", 205),
        (0.80, 0.1, "spearman", 269),
        (0.80, 0.1, "kendall", 93),
        (0.80, 0.2, "pearson", 56),
        (0.80, 0.2, "spearman", 72),
        (0.80, 0.2, "kendall", 27),
        (0.80, 0.3, "pearson", 28),
        (0.80, 0.3, "spearman", 35),
        (0.80, 0.3, "kendall", 15),
        (0.90, 0.1, "pearson", 62),
        (0.90, 0.1, "spearman", 86),
        (0.90, 0.1, "kendall", 30),
        (0.90, 0.2, "pearson", 20),
        (0.90, 0.2, "spearman", 27),
        (0.90, 0.3, "pearson", 12),
        (0.90, 0.3, "spearman", 16),
        (0.90, 0.3, "kendall", 8),
    ]
    assert len(table) == 67
    for theta, w, m, n in table:
        assert correlation_width_sample_size(theta, w, m)["n"] == n, (theta, w, m)


def test_cells_the_table_prints_differently():
    got = [
        correlation_width_sample_size(t, w, m)["n"]
        for t, w, m in (
            (0.1, 0.1, "spearman"),
            (0.1, 0.2, "spearman"),
            (0.1, 0.1, "kendall"),
            (0.4, 0.1, "kendall"),
            (0.9, 0.2, "kendall"),
        )
    ]
    assert got == [1515, 379, 662, 478, 12]


def test_exact_is_minimal_and_two_stage_close():
    for t, w, m in ((0.5, 0.2, "pearson"), (0.3, 0.1, "spearman"), (0.8, 0.3, "kendall")):
        r = correlation_width_sample_size(t, w, m)
        b = 4 if m == "kendall" else 3
        assert r["width_at_n"] <= w
        if r["n"] - 1 > b:
            import math

            c2 = {"pearson": 1.0, "spearman": 1 + t * t / 2, "kendall": 0.437}[m]
            h = 1.959963984540054 * math.sqrt(c2) / math.sqrt(r["n"] - 1 - b)
            assert math.tanh(math.atanh(t) + h) - math.tanh(math.atanh(t) - h) > w
        assert abs(correlation_width_sample_size(t, w, m, approach="two-stage")["n"] - r["n"]) <= 1


def test_two_stage_matches_the_table():
    # Bonett-Wright two-stage approximation, Table 7.85 cells
    for t, w, ns in ((0.5, 0.2, (219, 246, 99)), (0.8, 0.1, (205, 269, 93))):
        got = tuple(
            correlation_width_sample_size(t, w, m, approach="two-stage")["n"]
            for m in ("pearson", "spearman", "kendall")
        )
        assert got == ns
