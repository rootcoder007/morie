import pytest

from morie.fn.filtersim import _filters, filtersim

TI = [[1.0 if (x // 3) % 2 == 0 else 0.0 for x in range(24)] for y in range(24)]


def test_filters_match_definitions():
    offs, F = _filters(2)
    for k, (du, dv) in enumerate(offs):
        assert F[0][k] == pytest.approx(1 - abs(du) / 2, abs=1e-15)
        assert F[1][k] == pytest.approx(du / 2, abs=1e-15)
        assert F[2][k] == pytest.approx(abs(du) - 1, abs=1e-15)
        assert F[3][k] == pytest.approx(1 - abs(dv) / 2, abs=1e-15)


def test_pattern_count_classes_and_values():
    r = filtersim(TI, 10, 8, template=5, n_classes=4, seed=2)
    assert r.n_patterns == 20 * 20
    assert sum(r.class_sizes) == r.n_patterns
    vals = {v for row in r.realisation for v in row}
    assert vals <= {0.0, 1.0}
    assert all(v is not None for row in r.realisation for v in row)


def test_hard_data_honoured_and_deterministic():
    hd = {(2, 3): 0.0, (7, 1): 1.0}
    a = filtersim(TI, 10, 8, template=5, n_classes=4, seed=7, hard_data=hd)
    b = filtersim(TI, 10, 8, template=5, n_classes=4, seed=7, hard_data=hd)
    assert a.realisation == b.realisation
    assert a.realisation[3][2] == 0.0 and a.realisation[1][7] == 1.0
    with pytest.raises(ValueError):
        filtersim(TI, 5, 5, template=4)
