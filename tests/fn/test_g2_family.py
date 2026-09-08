"""Tests for the DL-shelf, graph and genomics batch of 18 modules.

Each test pins a property the paper states, not the implementation's
own output: an exact identity, a closed form, or a constructed case
with an independently known answer.
"""
import importlib
import math

import pytest


def M(name):
    return importlib.import_module("morie.fn." + name)


I2 = [[1.0, 0.0], [0.0, 1.0]]
CHAIN = {0: [1], 1: [0, 2], 2: [1]}


# --------------------------------------------------------------- grphmr
def test_grphmr_centrality_separates_hub_from_leaf():
    g = M("grphmr")
    adj = {0: [1], 1: [0, 2, 3], 2: [1], 3: [1]}
    r = g.centrality_encoding(adj, 4, [[float(d)] for d in range(4)])
    assert r["degrees"] == [1, 3, 1, 1]
    assert r["encoding"][1] != r["encoding"][0]


def test_grphmr_unreachable_is_tokenised_not_infinite():
    g = M("grphmr")
    sp = g.shortest_path_matrix({0: [1], 1: [0], 2: []}, 3)
    assert sp["distance"][0][2] == g.UNREACHABLE
    b = g.spatial_bias(sp["distance"], [0.0, -1.0])
    assert math.isfinite(b["bias"][0][2])


def test_grphmr_spatial_bias_discourages_without_masking():
    g = M("grphmr")
    adj = {0: [1], 1: [0, 2], 2: [1]}
    sp = g.shortest_path_matrix(adj, 3)
    bias = g.spatial_bias(sp["distance"], [0.0, -1.0, -6.0])
    H = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    w = g.graphormer_attention(H, I2, I2, I2,
                               bias["bias"])["weights"][0]
    assert abs(sum(w) - 1.0) < 1e-12
    assert 0.0 < w[2] < w[1]


def test_grphmr_edge_encoding_averages_along_the_path():
    g = M("grphmr")
    r = g.edge_encoding({(0, 2): [(0, 1), (1, 2)]},
                        {(0, 1): [1.0], (1, 2): [3.0]},
                        [[1.0], [1.0]])
    assert r["edge_bias"][(0, 2)] == pytest.approx(2.0)


# --------------------------------------------------------------- hetgnn
def _het():
    edges = {"m0": ["a0", "d0"], "m1": ["a0"], "m2": ["d0"],
             "a0": ["m0", "m1"], "d0": ["m0", "m2"]}
    types = {"m0": "M", "m1": "M", "m2": "M", "a0": "A", "d0": "D"}
    return edges, types


def test_hetgnn_relation_depends_on_the_metapath():
    h = M("hetgnn")
    edges, types = _het()
    mam = h.metapath_neighbours(edges, types, ["M", "A", "M"])
    mdm = h.metapath_neighbours(edges, types, ["M", "D", "M"])
    assert mam["neighbours"]["m0"] == ["m1"]
    assert mdm["neighbours"]["m0"] == ["m2"]


def test_hetgnn_both_attentions_normalise():
    h = M("hetgnn")
    H = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    na = h.node_attention(H[0], [1, 2], H, [0.5] * 4, I2)
    assert sum(na["alpha"]) == pytest.approx(1.0)
    Z = {"MAM": [[1.0, 0.0]] * 2, "MDM": [[0.0, 0.1]] * 2}
    sa = h.semantic_attention(Z, I2, [0.0, 0.0], [1.0, 0.0])
    assert sum(sa["beta"].values()) == pytest.approx(1.0)
    assert sa["beta"]["MAM"] > sa["beta"]["MDM"]


def test_hetgnn_inapplicable_metapath_raises():
    h = M("hetgnn")
    with pytest.raises(ValueError):
        h.node_attention([1.0, 0.0], [], [[1.0, 0.0]], [0.5] * 4, I2)


# ---------------------------------------------------------------- dmlqs
def test_dmlqs_exclusion_removes_every_totter():
    d = M("dmlqs")
    assert d.count_totters(CHAIN, 3, exclude_reverse=False)["totters"] > 0
    assert d.count_totters(CHAIN, 3, exclude_reverse=True)["totters"] == 0


def test_dmlqs_message_excludes_the_reverse_edge_exactly():
    d = M("dmlqs")
    h0 = {(0, 1): [1.0], (1, 0): [2.0], (1, 2): [4.0], (2, 1): [8.0]}
    ex = d.dmpnn_message_pass(h0, CHAIN, T=1)["edge_states"]
    inc = d.dmpnn_message_pass(h0, CHAIN, T=1,
                               exclude_reverse=False)["edge_states"]
    assert ex[(0, 1)][0] == pytest.approx(1.0)
    assert inc[(0, 1)][0] == pytest.approx(3.0)


def test_dmlqs_descriptors_are_concatenated_not_replaced():
    d = M("dmlqs")
    r = d.concat_descriptors([1.0, 2.0], [9.0])
    assert r["representation"] == [1.0, 2.0, 9.0]
    assert r["learned_dim"] == 2 and r["descriptor_dim"] == 1


# --------------------------------------------------------------- blip2v
def test_blip2v_output_width_is_the_query_count():
    b = M("blip2v")
    Q = b.query_tokens(8, 2, seed=1)
    small = [[float(i % 3), 1.0] for i in range(4)]
    large = [[float(i % 3), 1.0] for i in range(64)]
    a = b.qformer_attend(Q, small, I2, I2, I2)
    c = b.qformer_attend(Q, large, I2, I2, I2)
    assert len(a["output"]) == len(c["output"]) == 8
    assert c["compression"] == pytest.approx(8.0)


def test_blip2v_trainable_fraction_is_small():
    b = M("blip2v")
    r = b.trainable_fraction(188e6, 1.0e9, 7.0e9)
    assert r["fraction"] < 0.03
    assert r["fraction"] + r["frozen_fraction"] == pytest.approx(1.0)


def test_blip2v_image_text_score_is_the_max_query():
    b = M("blip2v")
    r = b.stage_one_objectives([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
    assert r["image_text_similarity"] == pytest.approx(1.0)
    assert r["best_query"] == 0


def test_blipqf_is_a_shim_not_a_copy():
    assert M("blipqf").qformer_attend is M("blip2v").qformer_attend


# ---------------------------------------------------------------- llavx
def test_llavx_stages_freeze_different_parts():
    lv = M("llavx")
    assert lv.training_stage(1)["trainable"] == ["projection"]
    assert "language_model" in lv.training_stage(1)["frozen"]
    assert lv.training_stage(2)["trainable"] == ["projection",
                                                 "language_model"]


def test_llavx_symbolic_representation_is_text_only():
    lv = M("llavx")
    r = lv.symbolic_representation(["a dog"], [("dog", 0, 0, 1, 1)])
    assert "dog" in r["text"] and "cat" not in r["text"]


def test_llavx_projection_mismatch_raises():
    lv = M("llavx")
    with pytest.raises(ValueError):
        lv.build_sequence([[1.0, 2.0]], [[1.0, 2.0, 3.0]])


# --------------------------------------------------------------- nrfrad
def test_nrfrad_weights_match_the_closed_form():
    nr = M("nrfrad")
    sig, n = 2.0, 8
    ts = [i / float(n) for i in range(n)]
    r = nr.volume_render([sig] * n, [[1.0]] * n, ts)
    dt = 1.0 / n
    for i in range(n - 1):
        expect = math.exp(-sig * dt * i) * (1.0 - math.exp(-sig * dt))
        assert r["weights"][i] == pytest.approx(expect, abs=1e-12)


def test_nrfrad_encoding_width_and_negative_density():
    nr = M("nrfrad")
    assert len(nr.positional_encoding([0.1, 0.2, 0.3], L=4)) == 3 + 24
    with pytest.raises(ValueError):
        nr.volume_render([-1.0], [[1.0]], [0.0])


def test_nrfrad_view_dependent_density_is_caught():
    nr = M("nrfrad")
    dirs = [[1, 0, 0], [0, 1, 0]]
    good = nr.density_is_view_independent(
        lambda p, d: {"sigma": p[0] ** 2}, [1.0, 0, 0], dirs)
    bad = nr.density_is_view_independent(
        lambda p, d: {"sigma": p[0] ** 2 + d[0]}, [1.0, 0, 0], dirs)
    assert good["view_independent"] and not bad["view_independent"]


# --------------------------------------------------------------- gsplat
def test_gsplat_factorisation_is_always_psd():
    gs = M("gsplat")
    cov = gs.covariance_from_scale_rotation([3.0, 0.5, 1.0],
                                            [0.3, 0.7, -0.2, 0.6])
    assert gs.is_positive_semidefinite(cov["covariance"])["psd"]
    raw = [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert not gs.is_positive_semidefinite(raw)["psd"]


def test_gsplat_identity_rotation_gives_scale_squared():
    gs = M("gsplat")
    cov = gs.covariance_from_scale_rotation([2.0, 3.0, 4.0],
                                            [1.0, 0.0, 0.0, 0.0])
    vals = sorted(gs.is_positive_semidefinite(cov["covariance"])
                  ["eigenvalues"])
    assert vals == pytest.approx([4.0, 9.0, 16.0])


def test_gsplat_compositing_equals_volume_rendering():
    gs, nr = M("gsplat"), M("nrfrad")
    sig, n = 2.0, 8
    ts = [i / float(n) for i in range(n)]
    vr = nr.volume_render([sig] * n, [[1.0]] * n, ts)
    dt = 1.0 / n
    alphas = [1.0 - math.exp(-sig * dt)] * (n - 1) + [1.0]
    ac = gs.alpha_composite([[1.0]] * n, alphas)
    assert ac["colour"][0] == pytest.approx(vr["colour"][0], abs=1e-12)


def test_gsplat_density_control_routes_by_size_and_opacity():
    gs = M("gsplat")
    r = gs.adaptive_density_control([0.001, 0.001, 0.0, 0.001],
                                    [0.001, 0.5, 0.001, 0.002],
                                    [0.9, 0.9, 0.9, 0.001])
    assert (r["clone"], r["split"], r["prune"]) == ([0], [1], [3])


# --------------------------------------------------------------- gan_an
def _toy():
    return (lambda z: [z[0], z[0] * 2.0, z[0] * 3.0],
            lambda x: [x[0], x[1] - x[0], x[2] - x[1]])


def test_gan_an_normal_reconstructs_and_anomalous_does_not():
    ga = M("gan_an")
    gen, feat = _toy()
    good = ga.invert_to_latent(gen([0.4]), gen, feat, 1, steps=300)
    bad = ga.invert_to_latent([0.4, 0.8, -5.0], gen, feat, 1, steps=300)
    assert good["score"] < 1e-2
    assert bad["score"] > 50.0 * good["score"]


def test_gan_an_scalar_discriminator_output_is_refused():
    ga = M("gan_an")
    with pytest.raises(ValueError):
        ga.discrimination_loss([1.0], [2.0])


def test_gan_an_separation_can_report_failure():
    ga = M("gan_an")
    assert ga.score_separation([0.1, 0.2], [5.0, 6.0])["auc"] == 1.0
    assert not ga.score_separation([1.0, 2.0], [1.0, 2.0])["separated"]


# ---------------------------------------------------------------- comet
def test_comet_pooled_features_are_six_blocks():
    cm = M("comet")
    r = cm.pooled_features([1.0, 2.0], [0.5, 0.5], [1.0, 2.0])
    assert r["dim"] == 12
    assert all(v == 0.0 for v in r["hyp_ref_diff"])
    assert any(v > 0.0 for v in r["hyp_src_diff"])


def test_comet_kendall_tau_exact():
    cm = M("comet")
    assert cm.kendall_tau([1, 2, 3, 4], [1, 2, 3, 4])["tau"] == 1.0
    assert cm.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1])["tau"] == -1.0
    assert cm.kendall_tau([1, 2, 3, 4],
                          [2, 1, 3, 4])["tau"] == pytest.approx(4 / 6)


def test_comet_triplet_loss_zero_only_when_ranking_holds():
    cm = M("comet")
    good = cm.triplet_loss([1.0, 0.0], [9.0, 0.0], [1.0, 0.0],
                           [1.0, 0.0])
    bad = cm.triplet_loss([9.0, 0.0], [1.0, 0.0], [1.0, 0.0],
                          [1.0, 0.0])
    assert good["satisfied"] and bad["loss"] > 0.0


def test_comet_reference_free_uses_no_reference():
    cm = M("comet")
    r = cm.reference_free([1.0, 2.0], [0.5, 0.5], [[0.1] * 8])
    assert r["reference_used"] is False


# ---------------------------------------------------------------- ibpfa
def test_ibpfa_log_probability_matches_the_closed_form():
    ib = M("ibpfa")
    lp = ib.ibp_log_probability([[1, 0], [1, 1]], 3.0)
    hand = 2 * math.log(3.0) - 3.0 * 1.5
    for m in (2, 1):
        hand += (math.lgamma(2 - m + 1) + math.lgamma(m)
                 - math.lgamma(3))
    assert lp == pytest.approx(hand, abs=1e-12)


def test_ibpfa_expected_counts_are_two_different_quantities():
    ib = M("ibpfa")
    r = ib.expected_features(10, 3.0)
    assert r["expected_total_features"] == pytest.approx(
        3.0 * sum(1.0 / i for i in range(1, 11)))
    assert r["expected_per_object"] == 3.0


def test_ibpfa_simulation_matches_the_expectation():
    ib = M("ibpfa")
    tot = [ib.sample_ibp(10, 3.0, seed=s)["K"] for s in range(200)]
    mean = sum(tot) / len(tot)
    assert abs(mean - 3.0 * sum(1.0 / i
                                for i in range(1, 11))) < 0.7


def test_ibpfa_left_ordering_is_idempotent():
    ib = M("ibpfa")
    lo = ib.left_ordered_form([[0, 1], [1, 1]])
    assert ib.left_ordered_form(lo["Z"])["Z"] == lo["Z"]


# --------------------------------------------------------------- baynav
def test_baynav_determinant_matches_the_numerical_jacobian():
    bn = M("baynav")
    u, w, b = [0.4, -0.2, 0.1], [0.3, 0.5, -0.7], 0.2
    z0, h = [0.2, -0.4, 0.6], 1e-6
    J = []
    for i in range(3):
        up, dn = list(z0), list(z0)
        up[i] += h
        dn[i] -= h
        fu = bn.planar_flow(up, u, w, b)["z"]
        fd = bn.planar_flow(dn, u, w, b)["z"]
        J.append([(fu[a] - fd[a]) / (2 * h) for a in range(3)])
    det = (J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
           - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
           + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]))
    assert bn.planar_flow(z0, u, w, b)["det"] == pytest.approx(det,
                                                               abs=1e-6)


def test_baynav_invertibility_constraint_binds():
    bn = M("baynav")
    r = bn.enforce_invertibility([-3.0, 0.0], [1.0, 0.0])
    assert r["adjusted"] and r["u_dot_w_after"] >= -1.0 - 1e-12


def test_baynav_depth_zero_is_mean_field():
    bn = M("baynav")
    assert bn.flow_log_density([0.1], -1.234, [])["log_q"] == -1.234


def test_baynav_transform_rejects_impossible_values():
    bn = M("baynav")
    with pytest.raises(ValueError):
        bn.transform_to_real(-1.0, "positive")
    r = bn.transform_to_real(4.0, "positive")
    assert r["real"] == pytest.approx(math.log(4.0))


# -------------------------------------------------------------- farmlmm
def _gwas(seed=7, n=60, p=12, causal=3):
    rng = M("_array_core").random.default_rng(seed)
    G = [[float(int(rng.uniform() * 3)) for _ in range(p)]
         for _ in range(n)]
    y = [2.0 * G[i][causal] + 0.3 * (float(rng.uniform()) - 0.5)
         for i in range(n)]
    return G, y


def test_farmlmm_selected_kinship_drops_the_confounding():
    fl = M("farmlmm")
    G, _ = _gwas()
    c_all = abs(fl.confounding(G, fl.kinship_from_markers(G)["K"],
                               3)["correlation"])
    c_sel = abs(fl.confounding(G,
                               fl.kinship_from_markers(G, [0, 1])["K"],
                               3)["correlation"])
    assert c_all > 0.4
    assert c_sel < c_all / 2.0


def test_farmlmm_recovers_the_planted_marker():
    fl = M("farmlmm")
    G, y = _gwas()
    scan = fl.fixed_effect_scan(y, G)
    assert scan["p"].index(min(scan["p"])) == 3
    assert 3 in fl.farmcpu(y, G, max_iter=6)["selected"]


# -------------------------------------------------------------- phmmsr
def test_phmmsr_gumbel_at_mu_is_one_minus_one_over_e():
    ph = M("phmmsr")
    assert ph.gumbel_pvalue(10.0, 10.0, 0.7) == pytest.approx(
        1.0 - math.exp(-1.0), abs=1e-12)
    assert ph.gumbel_pvalue(30.0, 10.0, 0.7) < ph.gumbel_pvalue(
        20.0, 10.0, 0.7)


def test_phmmsr_striped_layout_is_a_permutation():
    ph = M("phmmsr")
    r = ph.striped_layout(10, 4)
    assert sorted(r["order"]) == list(range(10))
    assert r["segments"] == 3


def test_phmmsr_rescale_fires_only_near_the_floor():
    ph = M("phmmsr")
    assert not ph.sparse_rescale([1.0, 2.0])["rescaled"]
    r = ph.sparse_rescale([1e-40, 2e-40])
    assert r["rescaled"]
    assert math.exp(r["log_offset"]) == pytest.approx(r["factor"],
                                                      rel=1e-9)


# -------------------------------------------------------------- genemt
def _sets(seed=11, ngene=80):
    rng = M("_array_core").random.default_rng(seed)
    nm = [int(3 + 40 * float(rng.uniform())) for _ in range(ngene)]
    lens = [float(v) * 1000.0 for v in nm]
    z = [0.4 * math.log(v) + 0.2 * (float(rng.uniform()) - 0.5)
         for v in nm]
    mem = [1.0 if v > 20 else 0.0 for v in nm]
    return nm, lens, z, mem


def test_genemt_size_covariates_kill_the_spurious_enrichment():
    gm = M("genemt")
    nm, lens, z, mem = _sets()
    naive = gm.gene_set_regression(z, mem)
    cov = gm.gene_covariates(nm, lens)["covariates"]
    adj = gm.gene_set_regression(z, mem, cov)
    assert naive["p"] < 1e-6
    assert abs(adj["beta"]) < abs(naive["beta"]) / 3.0


def test_genemt_gene_statistic_is_analytic():
    gm = M("genemt")
    rng = M("_array_core").random.default_rng(5)
    G = [[float(int(rng.uniform() * 3)) for _ in range(6)]
         for _ in range(50)]
    y = [1.5 * G[i][0] + 0.2 * (float(rng.uniform()) - 0.5)
         for i in range(50)]
    r = gm.gene_statistic(y, G)
    assert r["p"] < 1e-6 and r["df1"] >= 1


def test_genemt_covariates_reject_impossible_gene_sizes():
    gm = M("genemt")
    with pytest.raises(ValueError):
        gm.gene_covariates([0.0], [1000.0])


# -------------------------------------------------------------- metabd
def _rc(s):
    return "".join({"A": "T", "C": "G", "G": "C", "T": "A"}[c]
                   for c in reversed(s))


def test_metabd_canonical_tnf_is_strand_invariant():
    mb = M("metabd")
    seq = "ACGGTTAACGATCGATTACGCAGGTTACA"
    assert (mb.tetranucleotide_frequency(seq)["frequency"]
            == mb.tetranucleotide_frequency(_rc(seq))["frequency"])
    assert (mb.tetranucleotide_frequency(seq, canonical=False)
            ["frequency"]
            != mb.tetranucleotide_frequency(_rc(seq), canonical=False)
            ["frequency"])


def test_metabd_single_sample_abundance_is_refused():
    mb = M("metabd")
    with pytest.raises(ValueError):
        mb.abundance_correlation([1.0], [2.0])
    one = mb.composite_distance([0.1, 0.2], [0.3, 0.1], [5.0], [5.0])
    assert one["effective_weight"] == 0.0
    assert one["abundance_usable"] is False


def test_metabd_purity_and_completeness_differ():
    mb = M("metabd")
    r = mb.purity_completeness([[0, 1, 2], [3]], ["A", "A", "B", "A"])
    assert r["per_bin"][1]["purity"] == pytest.approx(1.0)
    assert r["per_bin"][1]["completeness"] == pytest.approx(1 / 3)


def test_metabd_length_weight_floors_short_contigs():
    mb = M("metabd")
    assert mb.length_weight(1000.0)["below_minimum"]
    assert 0.0 < mb.length_weight(20000.0)["weight"] < 1.0


# -------------------------------------------------------------- impfun
def test_impfun_union_beats_intersection():
    imf = M("impfun")
    r = imf.merge_panels({"P1": ["s1", "s2", "x1"],
                          "P2": ["s1", "s3", "x2"]},
                         ["s1", "s2", "s3"])
    assert r["kept_by_union"] == 5 and r["kept_by_intersection"] == 1
    assert r["targets"] == ["x1", "x2"]


def test_impfun_no_scaffold_snp_raises():
    imf = M("impfun")
    with pytest.raises(ValueError):
        imf.merge_panels({"P1": ["x1"]}, ["s9"])


def test_impfun_copying_concentrates_on_the_match():
    imf = M("impfun")
    refs = [[0] * 6, [1] * 6, [0, 1, 0, 1, 0, 1]]
    post = imf.copying_model([1] * 6, refs)["posterior"][5]
    assert post.index(max(post)) == 1 and max(post) > 0.9


def test_impfun_info_and_concordance_edges():
    imf = M("impfun")
    assert imf.info_score([0.0, 0.0, 0.0])["info"] == 1.0
    assert imf.concordance([0.1, 1.9], [0.0, 2.0])["concordance"] == 1.0
    assert imf.concordance([0.1, 1.9], [2.0, 0.0])["concordance"] == 0.0


# -------------------------------------------------------------- ocrwit
def test_ocrwit_bbox_is_page_size_invariant():
    oc = M("ocrwit")
    assert (oc.normalise_bbox([10, 20, 30, 40], 100, 200)
            == oc.normalise_bbox([20, 40, 60, 80], 200, 400)
            == [100, 100, 300, 200])


def test_ocrwit_segment_boxes_are_shared_within_a_line():
    oc = M("ocrwit")
    r = oc.segment_layout_boxes([[0, 0, 10, 10], [12, 0, 20, 10],
                                 [0, 50, 30, 60]],
                                ["l0", "l0", "l1"], 100, 100)
    assert r["per_token"][0] == r["per_token"][1]
    assert r["per_token"][0] != r["per_token"][2]
    assert r["n_segments"] == 2


def test_ocrwit_alignment_excludes_masked_words():
    oc = M("ocrwit")
    boxes = [[0, 0, 25, 25], [75, 75, 100, 100]]
    full = oc.word_patch_alignment(boxes, [0], 100, 100, 4)
    assert full["labels"][0] == 1 and full["labels"][1] == 0
    part = oc.word_patch_alignment(boxes, [0], 100, 100, 4,
                                   masked_text=[0])
    assert 0 not in part["labels"] and part["n_examples"] == 1
    with pytest.raises(ValueError):
        oc.word_patch_alignment(boxes, [0], 100, 100, 4,
                                masked_text=[0, 1])


def test_ocrwit_mask_units_partitions_the_sequence():
    oc = M("ocrwit")
    r = oc.mask_units(100, rate=0.3, seed=3)
    assert sorted(r["masked"] + r["kept"]) == list(range(100))
    assert abs(r["rate"] - 0.3) < 0.05
