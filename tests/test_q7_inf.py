# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every type-7 quantile helper against R's quantile(type = 7) with infinite values.

rmoriebricklayer 1.1.1 fixed its quantile helpers: lo + g * (hi - lo) gave NaN beside an
infinite value and an infinity when the difference overflowed. scripts/infq_sweep.py found the
same flaw in the helpers below. The expected values are R 4.6.1's
unname(quantile(v, p, type = 7)), not this package's output.
"""

import importlib
import math

import pytest

INF = math.inf
# (sorted values, p, R's quantile(type = 7))
CASES = [
    ([1.0, 2.0, INF], 0.5, 2.0),
    ([1.0, 2.0, INF], 0.75, INF),
    ([-INF, 1.0, 2.0], 0.25, -INF),
    ([-INF, 1.0], 0.5, -INF),
    ([INF, INF], 0.5, INF),
    ([-1e308, 1e308], 0.5, 0.0),
    ([1.0, 2.0, 3.0, 4.0], 0.5, 2.5),
    ([1.0, 2.0, 3.0, 4.0], 0.1, 1.3),
]

HELPERS = [
    ("morie._bayes_scaling", "_quantile"),
    ("morie.capsule._drift", "_q7"),
    ("morie.fn._s03core", "quantile7"),
    ("morie.fn._s04core", "quantile7"),
    ("morie.fn._spdiag", "quantile7"),
    ("morie.fn.aipw", "_quantile7"),
    ("morie.fn.baytrace", "_quantile7"),
    ("morie.fn.deseq2", "_quantile"),
    ("morie.fn.econgeo", "_q7"),
    ("morie.fn.efa2", "_quantile7"),
    ("morie.fn.kbwrt", "_q7"),
    ("morie.fn.mdsext", "_quantile7"),
    ("morie.fn.mkcorr", "_quantile7"),
    ("morie.fn.noiseacoustics", "_q7"),
    ("morie.fn.pibmd", "_quantile"),
    ("morie.fn.qmDS", "_quantile_interp"),
    ("morie.fn.seqsim", "_q7"),
    ("morie.fn.sgmci", "_q7"),
    ("morie.fn.spboot", "_q7"),
    ("morie.fn.vgextra", "_q7"),
    ("morie.fn.wildlife", "_quantile7"),
    ("morie.mrm_stockflow", "_quantile_type7"),
]


@pytest.mark.parametrize(("mod", "name"), HELPERS)
def test_type7_helper_matches_r_with_infinite_values(mod, name):
    f = getattr(importlib.import_module(mod), name)
    for v, p, want in CASES:
        got = float(f(list(v), p))
        if math.isinf(want):
            assert got == want, (name, v, p, got)
        else:
            assert got == pytest.approx(want, rel=1e-15, abs=1e-15), (name, v, p, got)
