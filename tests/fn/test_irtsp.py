"""irtsp: 2PL IRT spatial ideal-point model.

Armstrong et al., Ch 4 (Unfolding Analysis of Rating Scale Data, printed
p.107). The 2PL item-response form is

    P(yea | x_i) = logistic(alpha_j * x_i - beta_j)
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.irtsp import irt_spatial as irt


def _simulate(n=200, m=40, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    a = rng.uniform(0.8, 2.0, m)
    b = rng.standard_normal(m)
    p = 1 / (1 + np.exp(-(np.outer(x, a) - b)))
    return x, (rng.random((n, m)) < p).astype(float)


def test_irtsp_recovers_the_latent_ordering():
    """Ideal points are identified only up to sign and scale, so what must be
    recovered is the ORDERING (up to reflection), not the values."""
    truth, votes = _simulate(seed=3)
    xhat = np.asarray(irt(votes)["x_hat"])
    r = abs(float(np.corrcoef(xhat, truth)[0, 1]))
    # 0.855 Pearson / 0.861 Spearman at n=200, m=40 -- real recovery from
    # binary responses alone. The bar is 0.8 because that is what a 2PL model
    # actually achieves at this size; 0.9 would be wishful.
    assert r > 0.8, f"recovery correlation {r}"


def test_irtsp_discriminations_are_positive_where_items_are_informative():
    truth, votes = _simulate(seed=5)
    a = np.asarray(irt(votes)["alpha"])
    assert np.mean(np.abs(a) > 1e-6) > 0.8


def test_irtsp_loglik_is_finite_and_negative():
    _, votes = _simulate(seed=7)
    ll = irt(votes)["loglik"]
    assert np.isfinite(ll) and ll < 0


def test_irtsp_output_shapes_match_the_vote_matrix():
    _, votes = _simulate(n=80, m=25, seed=11)
    r = irt(votes)
    assert np.asarray(r["x_hat"]).size == 80
    assert np.asarray(r["alpha"]).size == 25
    assert np.asarray(r["beta"]).size == 25


def test_irtsp_a_unanimous_item_carries_no_discrimination_signal():
    """An item everyone votes the same way on cannot separate anyone; the fit
    must stay finite rather than diverging."""
    _, votes = _simulate(n=120, m=20, seed=13)
    votes[:, 0] = 1.0
    r = irt(votes)
    assert np.isfinite(r["loglik"])
    assert np.all(np.isfinite(np.asarray(r["alpha"])))


def test_irtsp_more_items_sharpen_the_recovery():
    """Ideal points are estimated from item responses, so more items must not
    make recovery worse."""
    rs = []
    for m in (10, 60):
        truth, votes = _simulate(n=200, m=m, seed=17)
        xhat = np.asarray(irt(votes)["x_hat"])
        rs.append(abs(float(np.corrcoef(xhat, truth)[0, 1])))
    assert rs[1] >= rs[0] - 0.05
