"""chronos / timesfm / timesf / momento.

Sources: Ansari, A. F. et al. (2024) TMLR, arXiv:2403.07815; Das, A.,
Kong, W., Sen, R. & Zhou, Y. (2024) ICML PMLR 235, arXiv:2310.10688;
Goswami, M. et al. (2024) ICML PMLR 235, arXiv:2402.03885."""
import math

import pytest

from morie.fn import _array_core as np
from morie.fn import timesf
from morie.fn.chronos import (dequantize, detokenize,
                              forecast_summary, mean_scale,
                              quantile_bins, quantize, tokenize,
                              uniform_bins)
from morie.fn.momento import (harmonise, mask_patches, masked_loss,
                              reconstruction_curve, task_mask)
from morie.fn.timesfm import (causal_mask, horizon_plan,
                              input_patches, rollout, rollout_steps)

B = uniform_bins(-5.0, 5.0, n_bins=21)


def test_mean_scaling_preserves_zeros():
    s = mean_scale([0.0, 4.0, 0.0, 8.0])
    assert s["scaled"][0] == 0.0 and s["scaled"][2] == 0.0


def test_the_scale_is_the_mean_absolute_value():
    x = [1.0, -3.0, 5.0]
    assert mean_scale(x)["scale"] == pytest.approx(3.0, abs=1e-12)


def test_an_all_zero_context_is_flagged_degenerate():
    assert mean_scale([0.0, 0.0, 0.0])["degenerate"]


def test_bin_edges_are_midway_between_centres():
    for i in range(len(B["edges"])):
        assert B["edges"][i] == pytest.approx(
            0.5 * (B["centers"][i] + B["centers"][i + 1]), abs=1e-12)


def test_a_bin_centre_round_trips_exactly():
    back = dequantize(quantize(B["centers"], B)["tokens"], B)
    for i in range(len(back)):
        assert back[i] == pytest.approx(B["centers"][i], abs=1e-12)


def test_the_round_trip_error_is_bounded_by_half_a_bin():
    half = 0.5 * (B["centers"][1] - B["centers"][0])
    rng = np.random.default_rng(1)
    v = [-5.0 + 10.0 * float(rng.uniform()) for _ in range(200)]
    rt = dequantize(quantize(v, B)["tokens"], B)
    assert max(abs(rt[i] - v[i]) for i in range(len(v))) <= half + 1e-12


def test_a_stationary_series_is_not_clipped():
    q = quantize([0.5 * math.sin(i / 4.0) for i in range(100)], B)
    assert q["in_range"]


def test_a_trending_series_is_clipped():
    q = quantize([0.2 * i for i in range(100)], B)
    assert q["clipped_fraction"] > 0.5


def test_the_vocabulary_includes_pad_and_eos():
    t = tokenize([1.0, 2.0], B)
    assert t["vocab_size"] == B["n_bins"] + 2
    assert t["tokens"][-1] == -2


def test_detokenising_restores_the_scale():
    t = tokenize([2.0, 4.0, 6.0], B)
    r = detokenize(t["tokens"], B, t["scale"])
    assert len(r) == 3
    assert r[0] == pytest.approx(2.0, abs=0.6)


def test_forecast_summary_normalises_the_distribution():
    p = [1.0] * 21
    s = forecast_summary(p, B)
    assert s["mean"] == pytest.approx(0.0, abs=1e-9)


def test_too_few_bins_is_refused():
    with pytest.raises(ValueError):
        uniform_bins(0.0, 1.0, 1)


def test_an_out_of_vocabulary_token_is_refused():
    with pytest.raises(ValueError):
        dequantize([500], B)


def test_too_few_samples_for_quantile_bins_is_refused():
    with pytest.raises(ValueError):
        quantile_bins([1.0, 2.0, 3.0], 50)


# ---------------------------------------------------------- timesfm
def test_history_is_padded_on_the_left():
    p = input_patches([float(i) for i in range(10)], 4)
    assert p["patches"][-1][-1] == 9.0
    assert p["n_padded"] == 2


def test_the_causal_mask_is_lower_triangular():
    m = causal_mask(4)["mask"]
    for i in range(4):
        for j in range(4):
            assert m[i][j] == (1.0 if j <= i else 0.0)


def test_rollout_steps_is_the_ceiling_of_the_ratio():
    assert rollout_steps(96, 32)["steps"] == 3
    assert rollout_steps(97, 32)["steps"] == 4


def test_a_long_output_patch_gives_a_single_step():
    r = rollout_steps(96, 128)
    assert r["steps"] == 1 and r["single_step"]


def test_a_longer_output_patch_cuts_the_step_count():
    h = horizon_plan(96, 32, 96)
    assert h["steps_asymmetric"] < h["steps_symmetric"]


def test_equal_patches_reproduce_the_symmetric_case():
    h = horizon_plan(96, 32, 32)
    assert h["steps_asymmetric"] == h["steps_symmetric"]


def test_the_rollout_returns_exactly_the_horizon():
    r = rollout([1.0] * 16, lambda p: [0.0] * 8, 20, 16, 8)
    assert len(r["forecast"]) == 20


def test_a_wrong_length_prediction_is_refused():
    with pytest.raises(ValueError):
        rollout([1.0] * 16, lambda p: [0.0] * 5, 8, 16, 8)


def test_a_zero_horizon_is_refused():
    with pytest.raises(ValueError):
        rollout_steps(0, 4)


def test_timesf_re_exports_timesfm():
    assert timesf.rollout_steps(50, 10) == rollout_steps(50, 10)


# ---------------------------------------------------------- momento
def batch():
    s = [[float(i), float(i) * 2.0] for i in range(32)]
    return harmonise([s], patch_len=8)


def test_each_channel_becomes_its_own_row():
    h = batch()
    assert h["n_series"] == 2
    assert h["n_patches"] == 4


def test_series_of_differing_channel_counts_share_a_batch():
    a = [[float(i)] for i in range(32)]
    b = [[float(i), float(i) * 2.0, float(i) * 3.0]
         for i in range(32)]
    h = harmonise([a, b], patch_len=8)
    assert h["n_series"] == 4


def test_masked_patches_become_zero():
    P = batch()["batch"][0]
    m = mask_patches(P, [1])
    assert all(v == 0.0 for v in m["masked"][1])


def test_visible_patches_are_untouched():
    P = batch()["batch"][0]
    m = mask_patches(P, [1])
    assert m["masked"][0] == P[0]


def test_masking_nothing_is_refused():
    P = batch()["batch"][0]
    with pytest.raises(ValueError):
        mask_patches(P, [])


def test_masking_everything_is_refused():
    P = batch()["batch"][0]
    with pytest.raises(ValueError):
        mask_patches(P, list(range(len(P))))


def test_the_loss_counts_only_masked_positions():
    P = batch()["batch"][0]
    m = mask_patches(P, [1])
    rec = [list(P[i]) for i in range(len(P))]
    rec[0] = [99.0] * len(P[0])
    L = masked_loss(P, rec, m["mask"])
    assert L["mse"] == pytest.approx(0.0, abs=1e-12)
    assert L["n_scored"] == len(P[0])


def test_a_wrong_reconstruction_of_the_masked_part_scores_badly():
    P = batch()["batch"][0]
    m = mask_patches(P, [1])
    rec = [list(P[i]) for i in range(len(P))]
    rec[1] = [99.0] * len(P[1])
    assert masked_loss(P, rec, m["mask"])["mse"] > 100.0


def test_forecast_masks_the_tail():
    assert task_mask(8, "forecast", 2) == [6, 7]


def test_imputation_masks_an_interior_gap():
    m = task_mask(8, "impute", 2)
    assert m[0] > 0 and m[-1] < 7


def test_an_unknown_task_is_refused():
    with pytest.raises(ValueError):
        task_mask(8, "regress", 2)


def test_a_span_covering_everything_is_refused():
    with pytest.raises(ValueError):
        task_mask(8, "forecast", 8)


def test_the_reconstruction_curve_reports_one_point_per_rate():
    P = batch()["batch"][0]
    c = reconstruction_curve(
        P, lambda mk, mask: [list(p) for p in mk], [0.25, 0.5])
    assert len(c["curve"]) == 2


def test_an_empty_series_list_is_refused():
    with pytest.raises(ValueError):
        harmonise([], 8)


def test_a_series_shorter_than_a_patch_is_refused():
    with pytest.raises(ValueError):
        harmonise([[[1.0], [2.0]]], 8)
