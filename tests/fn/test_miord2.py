"""Tests for miord2 (MICE with the normal model).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.miord2 import miord2


def _data_with_holes():
    # y roughly 2*x, one hole in each column
    rows = []
    for i in range(20):
        x = float(i)
        rows.append([x, 2.0 * x + (1.0 if i % 3 else -1.0)])
    rows[5][1] = None
    rows[11][0] = None
    return rows


def test_every_hole_is_filled_in_every_imputation():
    data = _data_with_holes()
    res = miord2(data, m=3, maxit=3, seed=1)
    assert res["m"] == 3
    assert len(res["imputations"]) == 3
    for imp in res["imputations"]:
        for row in imp:
            assert all(v is not None and v == v for v in row)


def test_observed_values_are_left_alone():
    data = _data_with_holes()
    res = miord2(data, m=2, maxit=2, seed=1)
    for imp in res["imputations"]:
        for i, row in enumerate(data):
            for j, v in enumerate(row):
                if v is not None:
                    assert abs(imp[i][j] - v) < 1e-12


def test_the_missing_mask_marks_exactly_the_holes():
    data = _data_with_holes()
    res = miord2(data, m=1, maxit=2, seed=1)
    mask = res["missing_mask"]
    assert mask[5][1] is True and mask[11][0] is True
    assert mask[0][0] is False


def test_imputations_differ_from_one_another():
    # proper multiple imputation draws, not one deterministic fill
    data = _data_with_holes()
    res = miord2(data, m=4, maxit=3, seed=1)
    vals = [imp[5][1] for imp in res["imputations"]]
    assert len(set(round(v, 9) for v in vals)) > 1


def test_the_fills_respect_the_relationship_in_the_data():
    data = _data_with_holes()
    res = miord2(data, m=5, maxit=5, seed=1)
    # row 5 has x = 5, so y should land near 10
    fills = [imp[5][1] for imp in res["imputations"]]
    assert abs(sum(fills) / len(fills) - 10.0) < 3.0


def test_seed_reproducibility():
    data = _data_with_holes()
    a = miord2(data, m=2, maxit=2, seed=7)["imputations"]
    b = miord2(data, m=2, maxit=2, seed=7)["imputations"]
    assert a == b


def test_validation():
    for call in (lambda: miord2([[1.0, 2.0], [3.0, 4.0]]),
                 lambda: miord2([[1.0, 2.0], [3.0], [1.0, 2.0]]),
                 lambda: miord2([[None, 1.0], [None, 2.0],
                                 [None, 3.0]])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
