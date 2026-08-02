"""Armstrong Sec 5.3.5 fit statistics (clfrt, prech, gmpre, agrmt, rollc)
against the p.143 footnote formulas, hand-computable fixtures."""

from morie.fn import _array_core as np
import pytest

from morie.fn.agrmt import agreement_score
from morie.fn.gmpre import geometric_mean_probability
from morie.fn.prech import proportional_reduction_error
from morie.fn.rollc import roll_call_analysis


def test_prech_apre_footnote_formula():
    # One roll call, 10 legislators: 6 yea, 4 nay (minority = 4).
    # Model misclassifies 1 of the 10; the null (all-modal) misses the 4
    # minority votes. APRE = (4 - 1) / 4 = 0.75 per the p.143 footnote.
    obs = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=float).T
    pred = obs.copy()
    pred[6, 0] = 1  # one nay predicted as yea
    out = proportional_reduction_error(obs, pred)
    assert out.value == pytest.approx(0.75)
    assert out.extra["null_errors"] == 4
    assert out.extra["model_errors"] == 1
    assert out.extra["classification_rate"] == pytest.approx(0.9)
    # a model no better than the null scores zero
    null_pred = np.ones_like(obs)
    assert proportional_reduction_error(obs, null_pred).value == pytest.approx(0.0)


def test_prech_aggregates_across_roll_calls():
    # Two roll calls with minorities 2 and 3; model errors 1 and 0.
    # APRE = ((2-1) + (3-0)) / (2+3) = 0.8.
    obs = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=float)
    obs = np.column_stack([np.array([1, 1, 1, 0, 0.0]), np.array([1, 1, 0, 0, 0.0])])
    # roll call 1: 3 yea 2 nay -> minority 2; roll call 2: 2 yea 3 nay -> minority 2
    pred = obs.copy()
    pred[3, 0] = 1  # one error on roll call 1
    out = proportional_reduction_error(obs, pred)
    assert out.extra["null_errors"] == 4
    assert out.extra["model_errors"] == 1
    assert out.value == pytest.approx(3 / 4)


def test_gmpre_exp_mean_loglik():
    # GMP = exp(mean log p(observed choice)); hand case with 4 cells.
    obs = np.array([[1.0, 0.0], [1.0, 1.0]])
    probs = np.array([[0.9, 0.2], [0.8, 0.6]])  # P(yea)
    # p(observed) = 0.9, 0.8 (nay: 1-0.2), 0.8, 0.6
    expected = np.exp(np.mean(np.log([0.9, 0.8, 0.8, 0.6])))
    out = geometric_mean_probability(obs, probs)
    assert out.value == pytest.approx(expected)
    # perfect certainty on every observed choice -> GMP = 1
    sure = np.where(obs == 1, 1.0, 0.0)
    assert geometric_mean_probability(obs, sure).value == pytest.approx(1.0, abs=1e-6)


def test_agrmt_sec_322_agreement_matrix():
    # Sec 3.2.2: agreement = shared votes cast the same way / shared votes.
    V = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 0, np.nan],
        ]
    )
    out = agreement_score(V)
    A = out.value["agreement_matrix"]
    assert A[0, 1] == pytest.approx(0.5)  # agree on votes 1 and 3 of 4
    assert A[0, 2] == pytest.approx(1.0)  # all three shared votes agree
    assert A[1, 2] == pytest.approx(2 / 3)
    assert out.extra["n_shared_votes"][0, 2] == 3
    with pytest.raises(ValueError):
        agreement_score(np.ones(5))


def test_rollc_descriptives():
    V = np.array(
        [
            [1, 1, 1, 0],
            [1, 0, np.nan, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=float,
    )
    out = roll_call_analysis(V)
    assert out.value["participation_rates"][1] == pytest.approx(0.75)
    assert out.value["yea_rates"][3] == pytest.approx(1.0)
    assert out.value["vote_margins"][0] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        roll_call_analysis(np.ones(3))
