"""Tests for the vision / self-supervised / recommender batch (19).

Each test pins a property the source paper states -- an exact identity,
a closed form, a printed figure, or a constructed case with a known
answer -- not the implementation's own output.
"""
import importlib
import math

import pytest


def M(name):
    return importlib.import_module("morie.fn." + name)


# ---------------------------------------------------------------- samseg
def test_samseg_prompt_type_changes_the_token():
    ss = M("samseg")
    te = {"foreground": [1.0] * 8, "background": [-1.0] * 8}
    fg = ss.encode_point_prompt([(0.3, 0.4)], [1], type_embeddings=te)
    bg = ss.encode_point_prompt([(0.3, 0.4)], [0], type_embeddings=te)
    assert fg["tokens"][0] != bg["tokens"][0]


def test_samseg_encoder_cost_amortises():
    ss = M("samseg")
    one = ss.amortised_cost(450.0, 50.0, 1)
    many = ss.amortised_cost(450.0, 50.0, 20)
    assert many["total_ms"] == pytest.approx(450.0 + 20 * 50.0)
    assert many["per_prompt_ms"] < one["per_prompt_ms"] / 4.0


def test_samseg_dense_prompt_is_summed():
    ss = M("samseg")
    r = ss.encode_mask_prompt([[1.0, 0.0], [0.0, 1.0]],
                              [[5.0, 5.0], [5.0, 5.0]])
    assert r["embedding"] == [[6.0, 5.0], [5.0, 6.0]]
    assert r["sparse"] is False


def test_samseg_rejects_bad_prompts():
    ss = M("samseg")
    with pytest.raises(ValueError):
        ss.encode_box_prompt([1.0, 1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        ss.encode_point_prompt([(0.1, 0.1)], [7])


# ---------------------------------------------------------------- samdec
def test_samdec_two_way_updates_both():
    sd = M("samdec")
    P = [[1.0, 0.0], [0.0, 1.0]]
    I = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    r = sd.two_way_block(P, I)
    assert r["prompt_tokens"] != P and r["image_tokens"] != I
    assert all(sum(row) == pytest.approx(1.0)
               for row in r["prompt_to_image"])


def test_samdec_focal_modulation_is_closed_form():
    sd = M("samdec")
    r = sd.focal_loss([0.9, 0.9], [1.0, 1.0], gamma=2.0)
    assert all(v == pytest.approx(0.01) for v in r["modulating"])


def test_samdec_dice_exact():
    sd = M("samdec")
    assert sd.dice_loss([1.0, 1.0, 0.0], [1.0, 1.0, 0.0])["loss"] == 0.0
    assert sd.dice_loss([1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0])["dice"] == pytest.approx(
                            2 / 3)


def test_samdec_dynamic_head_uses_the_output_token():
    sd = M("samdec")
    r = sd.dynamic_mask_head([1.0, 0.0],
                             [[[1.0, 0.0], [0.0, 1.0]],
                              [[2.0, 0.0], [0.0, 2.0]]])
    assert r["logits"] == [[1.0, 0.0], [2.0, 0.0]]


# ---------------------------------------------------------------- sammkr
def _masks():
    whole = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    part = [[1, 1, 0], [1, 1, 0], [0, 0, 0]]
    sub = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    return whole, part, sub


def test_sammkr_single_output_must_average():
    mk = M("sammkr")
    whole, part, _ = _masks()
    assert mk.average_of_valid_masks([whole,
                                      part])["ambiguous_fraction"] > 0


def test_sammkr_min_loss_beats_the_mean():
    mk = M("sammkr")
    whole, part, sub = _masks()
    r = mk.min_loss_over_masks([whole, part, sub], part,
                               lambda p, t: 1.0 - mk.iou(p, t))
    assert r["index"] == 1 and r["loss"] == pytest.approx(0.0)
    assert r["mean_loss"] > r["loss"]


def test_sammkr_ranking_reports_its_own_regret():
    mk = M("sammkr")
    whole, part, sub = _masks()
    good = mk.rank_masks([whole, part, sub], [0.2, 0.95, 0.1],
                         target=part)
    bad = mk.rank_masks([whole, part, sub], [0.99, 0.1, 0.1],
                        target=part)
    assert good["correct"] and good["regret"] == 0.0
    assert not bad["correct"] and bad["regret"] > 0.0


def test_sammkr_requires_three_outputs_for_the_nesting_claim():
    mk = M("sammkr")
    whole, part, _ = _masks()
    with pytest.raises(ValueError):
        mk.whole_part_subpart([whole, part])


# ---------------------------------------------------------------- sam2vd
def test_sam2vd_prompted_memory_survives_recent_churn():
    s2 = M("sam2vd")
    b = s2.memory_bank(n_recent=2, m_prompted=1)
    b = s2.push_memory(b, 0, [1.0, 0.0], prompted=True)
    for t in range(1, 5):
        b = s2.push_memory(b, t, [0.0, float(t)])
    assert [e["frame"] for e in b["prompted"]] == [0]
    assert [e["frame"] for e in b["recent"]] == [3, 4]


def test_sam2vd_empty_memory_is_the_image_model():
    s2 = M("sam2vd")
    r = s2.memory_attention([2.0, 3.0], s2.memory_bank(), 0)
    assert r["features"] == [2.0, 3.0]
    assert r["attended"] is False


def test_sam2vd_temporal_embedding_skips_prompted_frames():
    s2 = M("sam2vd")
    e = {"frame": 0, "features": [1.0, 1.0]}
    assert not s2.temporal_embedding(dict(e, prompted=True),
                                     9)["embedded"]
    assert s2.temporal_embedding(dict(e, prompted=False),
                                 9)["embedded"]


def test_sam2vd_propagation_conditions_after_the_first_frame():
    s2 = M("sam2vd")
    r = s2.propagate([1.0, 2.0, 3.0], lambda f: [float(f)],
                     lambda x, p: x[0], prompts={0: "mask"})
    assert r["conditioned"] == [False, True, True]


# ---------------------------------------------------------------- sdxlcd
def test_sdxlcd_filtering_discards_what_conditioning_keeps():
    sx = M("sdxlcd")
    sizes = [(200.0, 200.0)] * 39 + [(600.0, 600.0)] * 61
    r = sx.discarded_fraction(sizes, 256)
    assert r["fraction"] == pytest.approx(0.39)
    assert r["kept_with_conditioning"] == 100


def test_sdxlcd_crop_zero_is_uncropped():
    sx = M("sdxlcd")
    assert sx.crop_conditioning(0, 0)["object_centred"]
    assert not sx.crop_conditioning(64, 32)["object_centred"]
    with pytest.raises(ValueError):
        sx.crop_conditioning(-1, 0)


def test_sdxlcd_buckets_hold_the_pixel_count():
    sx = M("sdxlcd")
    r = sx.aspect_ratio_buckets([1.0, 16 / 9, 9 / 16, 4 / 3])
    assert r["max_pixel_error"] < 0.05


def test_sdxlcd_condition_vector_matches_the_timestep_width():
    sx = M("sdxlcd")
    r = sx.condition_vector(512, 512, 0, 0,
                            timestep_embedding=[1.0] * 32)
    assert r["width"] == 32
    with pytest.raises(ValueError):
        sx.condition_vector(512, 512, 0, 0,
                            timestep_embedding=[1.0] * 8)


# ---------------------------------------------------------------- vqgenc
BOOK = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]


def test_vqgenc_quantisation_is_nearest_neighbour():
    ve = M("vqgenc")
    r = ve.quantize([[0.9, 0.1], [0.1, 0.9], [0.05, 0.05]], BOOK)
    assert r["indices"] == [1, 2, 0]


def test_vqgenc_straight_through_is_identity_backward():
    ve = M("vqgenc")
    r = ve.straight_through([0.9, 0.1], [1.0, 0.0], [0.3, -0.7])
    assert r["forward"] == [1.0, 0.0]
    assert r["backward"] == [0.3, -0.7]


def test_vqgenc_the_two_terms_move_different_things():
    ve = M("vqgenc")
    cb = ve.codebook_loss([0.9, 0.1], [1.0, 0.0])
    cm = ve.commitment_loss([0.9, 0.1], [1.0, 0.0], beta=0.25)
    assert cb["loss"] == pytest.approx(0.02)
    assert cm["loss"] == pytest.approx(0.005)
    assert cb["gradient_flows_to"] != cm["gradient_flows_to"]


def test_vqgenc_attention_saving_is_the_square():
    ve = M("vqgenc")
    r = ve.sequence_length(256, 256, 16)
    assert r["tokens"] == 256
    assert r["speedup"] == pytest.approx(256.0 ** 2)


# ---------------------------------------------------------------- vqgdec
def test_vqgdec_decode_inverts_quantise():
    ve, vd = M("vqgenc"), M("vqgdec")
    idx = ve.quantize([[0.9, 0.1], [0.1, 0.9]], BOOK)["indices"]
    assert vd.decode_indices(idx, BOOK)["codes"] == [BOOK[i]
                                                     for i in idx]
    with pytest.raises(ValueError):
        vd.decode_indices([99], BOOK)


def test_vqgdec_adaptive_weight_shrinks_on_a_large_gan_gradient():
    vd = M("vqgdec")
    small = vd.adaptive_weight(1.0, 0.01)
    large = vd.adaptive_weight(1.0, 100.0)
    assert large["lambda"] == pytest.approx(1.0 / (100.0 + 1e-6))
    assert small["lambda"] > large["lambda"] * 1000


def test_vqgdec_patch_discriminator_scores_patches():
    vd = M("vqgdec")
    img = [[float(i * 4 + j) for j in range(4)] for i in range(4)]
    assert vd.patch_discriminator(img, patch=2)["n_patches"] == 4
    with pytest.raises(ValueError):
        vd.patch_discriminator(img, patch=3)


def test_vqgdec_sliding_windows_cover_the_grid():
    vd = M("vqgdec")
    assert vd.sliding_windows(6, 6, 4, 2)["covers_everything"]


# ---------------------------------------------------------------- vidgen
VIDEO = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]


def test_vidgen_identity_temporal_attention_is_the_image_case():
    vg = M("vidgen")
    assert vg.temporal_attention(VIDEO, identity=True)["video"] \
        == VIDEO
    assert vg.temporal_attention(VIDEO, identity=False)["video"] \
        != VIDEO


def test_vidgen_spatial_attention_treats_frames_as_batch():
    vg = M("vidgen")
    joint = vg.spatial_attention(VIDEO)["video"]
    alone = [vg.spatial_attention([f])["video"][0] for f in VIDEO]
    assert joint == alone


def test_vidgen_factorised_cost():
    vg = M("vidgen")
    r = vg.attention_cost(16, 1024)
    assert r["joint"] == (16 * 1024) ** 2
    assert r["factorised"] == 16 * 1024 ** 2 + 1024 * 16 ** 2


def test_vidgen_guidance_touches_only_observed_frames():
    vg = M("vidgen")
    r = vg.reconstruction_guidance([[0.0, 0.0], [9.0, 9.0]],
                                   [[1.0, 1.0]], [0], weight=2.0)
    assert r["gradient"][0] == [4.0, 4.0]
    assert r["gradient"][1] == [0.0, 0.0]


# ---------------------------------------------------------------- yolovx
def test_yolovx_decode_inverts_encode():
    yx = M("yolovx")
    box = [10.0, 20.0, 50.0, 60.0]
    ltrb = yx.encode_box(box, 3, 3, stride=8.0)["ltrb"]
    assert yx.decode_box(ltrb, 3, 3, stride=8.0) == pytest.approx(box)


def test_yolovx_location_outside_the_box_is_not_positive():
    yx = M("yolovx")
    with pytest.raises(ValueError):
        yx.encode_box([10.0, 20.0, 50.0, 60.0], 0, 0, stride=8.0)


def test_yolovx_center_sampling_beats_one_positive():
    yx = M("yolovx")
    r = yx.center_sampling([10.0, 20.0, 50.0, 60.0], 10, 10,
                           stride=8.0)
    assert r["n_candidates"] > 1


def test_yolovx_simota_k_is_dynamic_and_disjoint():
    yx = M("yolovx")
    r = yx.simota_assign([[0.1, 0.2, 0.3, 5.0], [5.0, 5.0, 0.4, 0.5]],
                         [[0.9, 0.8, 0.7, 0.0], [0.0, 0.0, 0.6, 0.5]],
                         top_q=4)
    assert r["dynamic_k"] == [2, 1]
    assert not set(r["assignment"][0]) & set(r["assignment"][1])


# ---------------------------------------------------------------- dnvtwo
def test_dnvtwo_deduplication_keeps_the_original():
    dn = M("dnvtwo")
    r = dn.deduplicate([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert r["n_after"] == 2 and r["dropped"] == [(1, 0)]


def test_dnvtwo_koleo_punishes_clustering():
    dn = M("dnvtwo")
    spread = dn.koleo([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0],
                       [0.0, -1.0]])
    clumped = dn.koleo([[1.0, 0.0], [0.999, 0.045], [-1.0, 0.0],
                        [0.0, -1.0]])
    assert clumped["loss"] > spread["loss"]
    with pytest.raises(ValueError):
        dn.koleo([[1.0, 0.0]])


def test_dnvtwo_sinkhorn_normalises_rows():
    dn = M("dnvtwo")
    r = dn.sinkhorn_knopp([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
                          iterations=3)
    assert all(v == pytest.approx(1.0, abs=1e-6)
               for v in r["row_sums"])


def test_dnvtwo_teacher_is_sharper_than_student():
    dn = M("dnvtwo")
    r = dn.self_distillation_loss([1.0, 0.0, 0.0], [3.0, 0.0, 0.0])
    assert max(r["teacher"]) > max(r["student"])


# ---------------------------------------------------------------- opnclp
def test_opnclp_fit_recovers_a_known_power_law():
    oc = M("opnclp")
    C = [1e3, 1e4, 1e5, 1e6]
    E = [0.5 * c ** (-0.3) for c in C]
    fit = oc.fit_power_law(C, E)
    assert fit["alpha"] == pytest.approx(0.3, abs=1e-9)
    assert fit["beta"] == pytest.approx(0.5, abs=1e-9)


def test_opnclp_extrapolation_distance_is_reported():
    oc = M("opnclp")
    C = [1e3, 1e4, 1e5, 1e6]
    fit = oc.fit_power_law(C, [0.5 * c ** (-0.3) for c in C])
    far = oc.predict(fit, 1e8)
    near = oc.predict(fit, 1e5)
    assert far["extrapolation_decades"] == pytest.approx(2.0)
    assert near["interpolated"]


def test_opnclp_distributions_give_different_exponents():
    oc = M("opnclp")
    C = [1e3, 1e4, 1e5, 1e6]
    r = oc.compare_scaling(C, [0.5 * c ** (-0.3) for c in C],
                           C, [0.5 * c ** (-0.15) for c in C])
    assert not r["same_law"]
    assert r["alpha_gap"] == pytest.approx(0.15, abs=1e-9)


def test_opnclp_infonce_is_symmetric_and_minimised_when_aligned():
    oc = M("opnclp")
    aligned = oc.infonce([[1.0, 0.0], [0.0, 1.0]],
                         [[1.0, 0.0], [0.0, 1.0]])
    swapped = oc.infonce([[1.0, 0.0], [0.0, 1.0]],
                         [[0.0, 1.0], [1.0, 0.0]])
    assert aligned["loss"] < swapped["loss"]
    assert aligned["image_to_text"] == pytest.approx(
        aligned["text_to_image"])


# ---------------------------------------------------------------- infmax
def test_infmax_jsd_floor_is_minus_two_log_two():
    im = M("infmax")
    same = im.jsd_estimator([0.0] * 8, [0.0] * 8)
    sep = im.jsd_estimator([5.0] * 8, [-5.0] * 8)
    assert same["estimate"] == pytest.approx(-2 * math.log(2.0),
                                             abs=1e-12)
    assert sep["estimate"] > same["estimate"] + 1.3
    assert sep["estimate"] <= 0.0


def test_infmax_dv_is_wrecked_by_one_outlier():
    im = M("infmax")
    small = im.dv_estimator([1.0] * 8, [0.0] * 8)
    big = im.dv_estimator([1.0] * 8, [0.0] * 7 + [30.0])
    assert abs(big["estimate"] - small["estimate"]) > 20.0
    assert big["bounded"] is False


def test_infmax_local_sees_structure_the_global_cannot():
    im = M("infmax")

    def critic(g, p):
        return g[0] * sum(p)

    glob = [[1.0], [-1.0]]
    spread = [[[1.0], [1.0], [1.0]], [[-1.0], [-1.0], [-1.0]]]
    conc = [[[3.0], [0.0], [0.0]], [[-3.0], [0.0], [0.0]]]
    ga = im.global_objective(glob, [[1.0, 1.0, 1.0],
                                    [-1.0, -1.0, -1.0]], critic)
    gb = im.global_objective(glob, [[3.0, 0.0, 0.0],
                                    [-3.0, 0.0, 0.0]], critic)
    la = im.local_objective(glob, spread, critic)
    lb = im.local_objective(glob, conc, critic)
    assert ga["objective"] == pytest.approx(gb["objective"], abs=1e-12)
    assert la["objective"] > lb["objective"] + 0.2


def test_infmax_needs_negatives():
    im = M("infmax")
    with pytest.raises(ValueError):
        im.local_objective([[1.0]], [[[1.0]]], lambda g, p: 1.0)


# ------------------------------------------------------------------ sdne
def _chain():
    return [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]


def test_sdne_unweighted_loss_rewards_predicting_nothing():
    sn = M("sdne")
    A = _chain()
    zero = [[0.0] * 4 for _ in range(4)]
    half = [[0.5 if A[i][j] else 0.0 for j in range(4)]
            for i in range(4)]
    assert sn.second_order_loss(A, zero, beta=1.0)["loss"] > \
        sn.second_order_loss(A, half, beta=1.0)["loss"]
    assert sn.second_order_loss(A, zero, beta=5.0)["loss"] > \
        sn.second_order_loss(A, half, beta=5.0)["loss"]


def test_sdne_perfect_reconstruction_is_zero():
    sn = M("sdne")
    A = _chain()
    assert sn.second_order_loss(A, A, beta=5.0)["loss"] == \
        pytest.approx(0.0)


def test_sdne_first_order_is_zero_iff_linked_pairs_coincide():
    sn = M("sdne")
    A = _chain()
    assert sn.first_order_loss(A, [[0.0], [0.0], [0.0],
                                   [9.0]])["loss"] == \
        pytest.approx(0.0)
    assert sn.first_order_loss(A, [[0.0], [1.0], [0.0],
                                   [0.0]])["loss"] > 0.0


def test_sdne_second_order_pairs_dominate_in_a_star():
    sn = M("sdne")
    star = [[0] * 8 for _ in range(8)]
    for v in range(1, 8):
        star[0][v] = star[v][0] = 1
    r = sn.proximity_counts(star)
    assert r["first_order_pairs"] == 7
    assert r["second_order_pairs"] == 21


def test_sdne_rejects_beta_below_one():
    sn = M("sdne")
    with pytest.raises(ValueError):
        sn.penalty_matrix(_chain(), beta=0.5)


# ------------------------------------------------------------------ se3T
POS = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.5],
       [-1.0, 0.3, 1.0]]
T0 = [0.5, -0.2, 1.1, 0.3]
T1 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
      [1.0, 1.0, 1.0]]


def test_se3T_output_rotates_with_the_input():
    s3 = M("se3T")
    r = s3.check_equivariance(POS, T0, T1)
    assert r["type1_deviation"] < 1e-12
    assert r["type0_deviation"] < 1e-12
    assert r["weights_invariant"]


def test_se3T_check_catches_a_non_equivariant_layer():
    s3 = M("se3T")

    def broken(p, s, v):
        r = s3.se3_attention(p, s, v)
        return {"type1": [[x + p[i][0] for x in r["type1"][i]]
                          for i in range(len(p))],
                "type0": r["type0"], "weights": r["weights"]}

    assert not s3.check_equivariance(POS, T0, T1,
                                     layer=broken)["equivariant"]


def test_se3T_rotation_is_orthogonal():
    s3 = M("se3T")
    R = s3.rotation_matrix([0.0, 0.0, 1.0], math.pi / 2)
    for i in range(3):
        for j in range(3):
            want = 1.0 if i == j else 0.0
            got = sum(R[i][a] * R[j][a] for a in range(3))
            assert got == pytest.approx(want, abs=1e-12)


def test_se3T_radial_kernel_sees_only_distance():
    s3 = M("se3T")
    with pytest.raises(ValueError):
        s3.radial_kernel(-1.0)


# ---------------------------------------------------------------- ssmpar
PAIRS = [(0.5, 1.0), (0.9, -2.0), (1.1, 0.3), (0.7, 4.0),
         (0.95, -1.5), (1.05, 0.25), (0.6, 2.0)]


def test_ssmpar_parallel_equals_sequential():
    sp = M("ssmpar")
    seq = sp.sequential_scan(PAIRS, x0=0.25)["states"]
    par = sp.parallel_scan(PAIRS, x0=0.25)["states"]
    for i in range(len(PAIRS)):
        assert par[i] == pytest.approx(seq[i], abs=1e-12)


def test_ssmpar_composition_is_associative():
    sp = M("ssmpar")
    r = sp.check_associativity((0.5, 1.0), (0.9, -2.0), (1.1, 0.3))
    assert r["associative"]


def test_ssmpar_depth_is_logarithmic():
    sp = M("ssmpar")
    r = sp.scan_depth(1024)
    assert r["parallel_depth"] == 10
    assert r["sequential_depth"] == 1024


def test_ssmpar_empty_sequence_raises():
    sp = M("ssmpar")
    with pytest.raises(ValueError):
        sp.parallel_scan([])


# ------------------------------------------------------------------ dssm
def test_dssm_letter_trigrams_are_exact():
    ds = M("dssm")
    assert ds.letter_ngrams("good") == ["#go", "goo", "ood", "od#"]


def test_dssm_word_hashing_reduces_the_input_layer():
    ds = M("dssm")
    r = ds.collision_rate(["good", "goods", "dog", "god", "cat",
                           "cats", "act"])
    assert r["ngram_dimension"] > 0
    assert r["collision_rate"] < 0.5


def test_dssm_unseen_word_still_has_a_representation():
    ds = M("dssm")
    assert sum(ds.word_hash(["xyzzyx"])["vector"]) > 0


def test_dssm_posterior_normalises_and_gamma_flattens_it():
    ds = M("dssm")
    sharp = ds.click_posterior([1.0, 0.0], [1.0, 0.0],
                               [[0.0, 1.0], [-1.0, 0.0]], gamma=10.0)
    flat = ds.click_posterior([1.0, 0.0], [1.0, 0.0],
                              [[0.0, 1.0], [-1.0, 0.0]], gamma=0.001)
    assert sum(sharp["posterior"]) == pytest.approx(1.0)
    assert sharp["posterior_clicked"] > 0.99
    assert flat["posterior_clicked"] == pytest.approx(1 / 3, abs=1e-3)


def test_dssm_zero_vector_has_no_direction():
    ds = M("dssm")
    with pytest.raises(ValueError):
        ds.cosine_similarity([0.0, 0.0], [1.0, 0.0])


# ------------------------------------------------------------------ twoT
def test_twoT_correction_reverses_a_popularity_ranking():
    tt = M("twoT")
    r = tt.retrieve([1.0], [[0.80], [0.75]],
                    probabilities=[0.5, 0.01], temperature=1.0,
                    top_k=2)
    assert r["uncorrected_top_k"][0] == 0
    assert r["top_k"][0] == 1
    assert r["changed"]


def test_twoT_shift_is_minus_log_p():
    tt = M("twoT")
    r = tt.corrected_logits([0.8, 0.75], [0.5, 0.01])
    assert r["shift"][0] == pytest.approx(-math.log(0.5))
    assert r["shift"][1] == pytest.approx(-math.log(0.01))


def test_twoT_streaming_frequency_converges_to_one_over_gap():
    tt = M("twoT")
    hits = {t: [7] for t in range(0, 200, 4)}
    r = tt.streaming_frequency(hits, 200, alpha=0.2)
    assert r["probability"][7] == pytest.approx(0.25, abs=1e-4)


def test_twoT_towers_are_normalised():
    tt = M("twoT")
    r = tt.tower_embedding([3.0, 4.0], [[1.0, 0.0], [0.0, 1.0]])
    assert sum(v * v for v in r["embedding"]) == pytest.approx(1.0)
    assert r["norm"] == pytest.approx(5.0)


def test_twoT_rejects_impossible_probabilities():
    tt = M("twoT")
    with pytest.raises(ValueError):
        tt.corrected_logits([1.0], [0.0])


# ----------------------------------------------------------------- sse4r
def test_sse4r_same_history_different_user_differs():
    s4 = M("sse4r")
    hist = [[1.0, 0.0], [0.0, 1.0]]
    a = s4.predict_next(hist, [1.0], [[1.0, 0.0], [0.0, 1.0]])
    b = s4.predict_next(hist, [-1.0], [[1.0, 0.0], [0.0, 1.0]])
    assert a["scores"] != b["scores"]


def test_sse4r_user_vector_is_on_every_position():
    s4 = M("sse4r")
    r = s4.personalise([[1.0, 0.0], [0.0, 1.0]], [7.0, 8.0])
    assert all(row[-2:] == [7.0, 8.0] for row in r["sequence"])
    assert r["width"] == 4


def test_sse4r_p_zero_is_the_identity():
    s4 = M("sse4r")
    r = s4.sse_replace([1, 2, 3, 4], 10, p=0.0)
    assert r["indices"] == [1, 2, 3, 4] and r["rate"] == 0.0


def test_sse4r_observed_rate_is_p_times_one_minus_one_over_n():
    s4 = M("sse4r")
    r = s4.sse_replace([i % 20 for i in range(200)], 20, p=0.5,
                       seed=3)
    expect = s4.expected_replacement(0.5, 20)["expected_rate"]
    assert abs(r["rate"] - expect) < 0.08
    assert expect < 0.5


def test_sse4r_rejects_out_of_table_indices():
    s4 = M("sse4r")
    with pytest.raises(ValueError):
        s4.sse_replace([25], 20, p=0.5)


# ---------------------------------------------------------------- diffRC
def test_diffRC_zero_scale_is_the_identity():
    dr = M("diffRC")
    sch = dr.noise_schedule(10, scale=0.0)
    assert all(v == pytest.approx(1.0) for v in sch["alpha_bar"])
    out = dr.denoise([1.0, 2.0], lambda x, t: x, sch)
    assert out["x0"] == pytest.approx([1.0, 2.0], abs=1e-12)


def test_diffRC_reduced_scale_keeps_the_history():
    dr = M("diffRC")
    small = dr.noise_schedule(10, scale=0.01)
    full = dr.noise_schedule(10, scale=1.0)
    assert small["signal_retained"] > 0.99
    assert full["signal_retained"] < small["signal_retained"]


def test_diffRC_forward_matches_the_closed_form():
    dr = M("diffRC")
    r = dr.forward_corrupt([1.0, 4.0], 0.25)
    assert r["mean"] == [0.5, 2.0]
    assert r["std"] == pytest.approx(math.sqrt(0.75))
    with pytest.raises(ValueError):
        dr.forward_corrupt([1.0], 1.5)


def test_diffRC_importance_sampling_concentrates():
    dr = M("diffRC")
    imp = dr.importance_weights([0.0, 0.0, 0.0, 9.0], smoothing=0.0)
    uni = dr.importance_weights([0.0, 0.0, 0.0, 9.0], uniform=True)
    assert imp["weights"][3] == pytest.approx(1.0)
    assert uni["weights"][3] == pytest.approx(0.25)
    assert imp["effective_steps"] < uni["effective_steps"]
