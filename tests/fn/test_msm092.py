"""Verification tests for msm092.

Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
Statistical Machine Learning Methods for Genomic Prediction*,
Springer, ch 7, eq. 7.3 p.219, the ordinal latent predictor. Expected values are recomputed in the test body
from the equation the module cites.
"""

import math

import pytest

from morie.fn.msm092 import mvsml_bayesian_regression_pt2_eq_7_3


def test_the_latent_predictor_stacks_the_three_effect_blocks():
    # eq 7.3: L = X_E beta_E + X beta + X_EM beta_EM + eps
    X_E = [[1.0, 0.0], [0.0, 1.0]]
    X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    X_EM = [[1.0], [2.0]]
    res = mvsml_bayesian_regression_pt2_eq_7_3(2, X_E=X_E, X=X, X_EM=X_EM)
    assert res["estimate"] == pytest.approx(2 + 3 + 1, rel=1e-12)
    assert res["widths"] == {"environments": 2, "markers": 3,
                             "env_x_marker": 1}


def test_the_blocks_are_stacked_in_the_printed_order():
    res = mvsml_bayesian_regression_pt2_eq_7_3(1, X_E=[[7.0]], X=[[8.0, 9.0]], X_EM=[[10.0]])
    assert [list(r) for r in res["design"]] == [[7.0, 8.0, 9.0, 10.0]]


def test_the_marker_block_alone_is_a_valid_predictor():
    res = mvsml_bayesian_regression_pt2_eq_7_3(2, X=[[1.0, 2.0], [3.0, 4.0]])
    assert res["estimate"] == pytest.approx(2.0, rel=1e-12)
    assert res["widths"] == {"markers": 2}


def test_a_predictor_with_no_blocks_at_all_is_refused():
    with pytest.raises(ValueError):
        mvsml_bayesian_regression_pt2_eq_7_3(2)
