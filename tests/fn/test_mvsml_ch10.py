"""Known-answer tests for MVSML chapter 10 (ANN, backpropagation).

The decisive check is a central-difference gradient against the
analytic backpropagation gradients of eq. (10.10)-(10.16): if the
chain rule were wrong anywhere, they would disagree.
"""
import math

from morie.fn import _gp_core as gp
from morie.fn.msm245 import (mvsml_reproducing_kernel_eq_10_4,
                             mvsml_ann_forward)
from morie.fn.msm249 import mvsml_reproducing_kernel_eq_10_5
from morie.fn.msm247 import mvsml_reproducing_kernel_eq_10_6
from morie.fn.msm248 import mvsml_reproducing_kernel_eq_10_9
from morie.fn.msm246 import mvsml_reproducing_kernel_eq_10_10
from morie.fn.msm250 import mvsml_reproducing_kernel_eq_10_12
from morie.fn.msm253 import mvsml_reproducing_kernel_eq_10_14
from morie.fn.msm255 import mvsml_reproducing_kernel_eq_10_17

# XOR with a bias input fixed at 1 (p.409)
X = [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0],
     [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
Y = [[0.0], [1.0], [1.0], [0.0]]


def _net(seed=3):
    rng = gp.np.random.default_rng(seed)
    W1 = [[float(rng.normal(0, 1)) for _ in range(3)]
          for _ in range(4)]
    W2 = [[float(rng.normal(0, 1)) for _ in range(4)]]
    return [W1, W2]


def test_forward_pass_matches_the_definition():
    W = _net()
    r = mvsml_reproducing_kernel_eq_10_4(X, W, ["logistic",
                                                "identity"])
    # hand-compute the first hidden unit of the first pattern
    z = sum(W[0][0][p] * X[0][p] for p in range(3))
    v = 1.0 / (1.0 + math.exp(-z))
    assert abs(r["layers"][1][0][0] - v) < 1e-12
    # and the output
    out = sum(W[1][0][k] * r["layers"][1][0][k] for k in range(4))
    assert abs(r["output"][0][0] - out) < 1e-12


def test_eq_10_5_sse_matches_the_formula():
    yh = [[0.5], [0.2]]
    yy = [[1.0], [0.0]]
    r = mvsml_reproducing_kernel_eq_10_5(yh, yy)
    assert abs(r["sse"] - 0.5 * (0.25 + 0.04)) < 1e-12


def test_eq_10_6_and_10_9_layer_pieces():
    W = _net()
    h = mvsml_reproducing_kernel_eq_10_6(X, W[0])
    assert len(h["V"]) == 4 and len(h["V"][0]) == 4
    o = mvsml_reproducing_kernel_eq_10_9(h["V"], W[1])
    assert len(o["y_hat"]) == 4 and len(o["y_hat"][0]) == 1
    # identity output activation means y_hat == z (eq. 10.9)
    for i in range(4):
        assert abs(o["y_hat"][i][0] - o["z"][i][0]) < 1e-12


def test_backprop_gradients_match_central_differences():
    # this is the real check on eq. (10.10)-(10.16)
    W = _net(seed=11)
    acts = ["logistic", "identity"]
    ana = gp.ann_backprop_gradients(X, Y, W, acts)["gradients"]
    num = gp.ann_numeric_gradient(X, Y, W, acts)
    for li in range(len(W)):
        for u in range(len(ana[li])):
            for v in range(len(ana[li][u])):
                assert abs(ana[li][u][v] - num[li][u][v]) < 1e-6


def test_gradients_match_with_tanh_and_relu_too():
    W = _net(seed=5)
    for hidden in ("tanh", "relu"):
        acts = [hidden, "logistic"]
        ana = gp.ann_backprop_gradients(X, Y, W, acts)["gradients"]
        num = gp.ann_numeric_gradient(X, Y, W, acts)
        for li in range(len(W)):
            for u in range(len(ana[li])):
                for v in range(len(ana[li][u])):
                    assert abs(ana[li][u][v]
                               - num[li][u][v]) < 1e-5


def test_eq_10_10_weight_change_is_minus_eta_times_gradient():
    W = _net()
    r = mvsml_reproducing_kernel_eq_10_10(X, Y, W, eta=0.25)
    for li in range(len(W)):
        for u in range(len(r["gradients"][li])):
            for v in range(len(r["gradients"][li][u])):
                assert abs(r["weight_changes"][li][u][v]
                           + 0.25 * r["gradients"][li][u][v]) < 1e-12


def test_eq_10_12_and_10_14_select_the_right_layers():
    W = _net()
    out = mvsml_reproducing_kernel_eq_10_12(X, Y, W)
    hid = mvsml_reproducing_kernel_eq_10_14(X, Y, W)
    assert len(out["delta_w"]) == 1          # one output unit
    assert len(hid["delta_w"]) == 4          # four hidden units
    assert len(hid["delta_w"][0]) == 3       # three inputs


def test_training_decreases_the_loss_monotonically():
    # plain gradient descent (eq. 10.13/10.17) decreases the loss
    # monotonically only for a small enough learning rate; eta is the
    # step size "specified by the user" in eq. (10.10)
    W = _net(seed=7)
    r = mvsml_reproducing_kernel_eq_10_17(X, Y, W, eta=0.05,
                                          n_iter=3000,
                                          activations=["logistic",
                                                       "identity"])
    h = r["history"]
    assert h[-1] < h[0]
    for a, b in zip(h, h[1:]):
        assert b <= a + 1e-9                 # never increases
    assert r["loss"] < 0.35                  # XOR is being learned


def test_too_large_a_learning_rate_overshoots():
    # the flip side of eq. (10.10): eta scales the step, so an
    # over-large step overshoots the minimum and the loss rises
    W = _net(seed=7)
    big = mvsml_reproducing_kernel_eq_10_17(X, Y, W, eta=0.9,
                                            n_iter=50,
                                            activations=["logistic",
                                                         "identity"])
    assert max(big["history"]) > big["history"][0]


def test_bias_via_an_input_fixed_at_one():
    # p.409: the bias is accounted for by an extra input fixed at 1
    W = [[[2.0, 0.0, 0.0]], ]
    r = mvsml_reproducing_kernel_eq_10_4(X, W, ["identity"])
    assert all(abs(row[0] - 2.0) < 1e-12 for row in r["output"])


def test_canonical_alias():
    assert mvsml_ann_forward is mvsml_reproducing_kernel_eq_10_4
