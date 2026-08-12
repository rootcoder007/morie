"""Tests for droSPI (Standardized Precipitation Index, McKee 1993).

Replaces the generated stub, which imported ``spi``.
"""

from morie.fn.droSPI import droSPI


def _series(n_years=40, seed=3):
    st = [seed]

    def r():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    # a seasonal cycle plus noise, 12 months a year
    out = []
    for y in range(n_years):
        for m in range(12):
            base = 40.0 + 30.0 * (1 if m in (5, 6, 7) else 0)
            out.append(max(0.0, base * (0.4 + 1.2 * r())))
    return out


def test_the_index_is_standardised():
    res = droSPI(_series(), scale=3, by_month=False)
    vals = [v for v in res["spi"] if v is not None]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    assert abs(mean) < 0.2
    assert abs(var ** 0.5 - 1.0) < 0.2


def test_the_first_months_have_no_index_at_scale_3():
    res = droSPI(_series(), scale=3)
    # the accumulation window is not full yet, so there is no index; the
    # module says None rather than inventing a value
    assert res["spi"][0] is None and res["spi"][1] is None
    assert res["spi"][2] is not None
    assert res["totals"][0] is None


def test_a_dry_spell_scores_negative_and_a_wet_one_positive():
    x = _series()
    x[100:112] = [1.0] * 12                      # a drought year
    res = droSPI(x, scale=3, by_month=False)
    assert res["spi"][105] < -1.0
    wettest = max(range(len(res["spi"])),
                  key=lambda i: res["totals"][i]
                  if res["totals"][i] is not None else -1e18)
    assert res["spi"][wettest] > 0


def test_scale_changes_the_accumulation_window():
    x = _series()
    a = droSPI(x, scale=3)
    b = droSPI(x, scale=12)
    assert a["scale"] == 3 and b["scale"] == 12
    assert sum(1 for v in b["spi"] if v is None) > \
        sum(1 for v in a["spi"] if v is None)


def test_by_month_fits_a_distribution_per_calendar_month():
    res = droSPI(_series(), scale=3, by_month=True)
    assert res["by_month"] is True
    assert len(res["params"]) == 12


def test_validation():
    try:
        droSPI([-1.0] * 60)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
