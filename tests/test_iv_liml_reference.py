"""LIML against linearmodels.iv.IVLIML (unadjusted covariance, debiased).

Reference values: ``IVLIML(y, [const, w], [d], Z).fit(cov_type="unadjusted",
debiased=True)`` on the deterministic data below. The kappa used to be
the reciprocal of the LIML eigenvalue, giving beta_d = 3.04 for a true 1.5.
"""

import math

import pytest

from morie import iv
from morie.fn import _frame_core as pd

REF = {
    "over": (
        ["z1", "z2"],
        [1.001874679189008, 1.5024766246523928, -0.6999269712251613],
        [0.038113786441526035, 0.08136046445887914, 0.0631036039774749],
    ),
    "just": (
        ["z1"],
        [1.0018777268733108, 1.501063013836254, -0.6993561200081572],
        [0.0381511554008642, 0.09125564344355117, 0.06531681168048792],
    ),
}


def _data():
    rows = []
    for i in range(400):
        z1, z2, w = math.sin(1.1 * i), math.cos(0.7 * i + 0.3), math.sin(0.37 * i + 1.0)
        u = math.sin(2.9 * i + 0.5)
        d = 0.6 * z1 + 0.3 * z2 + 0.4 * w + 0.8 * u + 0.3 * math.cos(5.1 * i)
        y = 1.0 + 1.5 * d - 0.7 * w + u + 0.4 * math.sin(7.7 * i)
        rows.append({"y": y, "d": d, "z1": z1, "z2": z2, "w": w})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("case", sorted(REF))
def test_liml_matches_linearmodels(case):
    zs, coef, se = REF[case]
    r = iv.liml(_data(), "y", ["d"], zs, ["w"], robust=False)
    for got, ref in zip(r.coefficients, coef):
        assert abs(float(got) - ref) <= 1e-10 * abs(ref)
    for got, ref in zip(r.std_errors, se):
        assert abs(float(got) - ref) <= 1e-10 * ref


def test_just_identified_liml_is_2sls():
    d = _data()
    a = iv.liml(d, "y", ["d"], ["z1"], ["w"], robust=False)
    b = iv.tsls(d, "y", ["d"], ["z1"], ["w"], robust=False)
    for x, y in zip(a.coefficients, b.coefficients):
        assert abs(float(x) - float(y)) <= 1e-10 * max(1.0, abs(float(y)))
