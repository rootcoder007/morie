import json, math
import numpy as np
from morie.fn import (hma2c, hma3c, hmadab, hmadam, hmadgr, hmadmw, hmadmx, hmaen,
                      hmagc, hmagrd, hmaic, hmalbt, hmalex, hmanae, hmarim, hmauc,
                      hmauxpt, hmbag, hmbart, hmbat, hmbdn, hmbel, hmbert, hmbf16,
                      hmbftn, hmbgdg, hmbic, hmbin, hmblip, hmblp2, hmbms, hmbnm,
                      hmbntr, hmbp, hmbpet, hmbrch, hmbrnn, hmbrob, hmbsz, hmbv,
                      gr1cy, grac, grada2, gradaw, gradmo, grael, graic, grarma,
                      grauc, graut, grbag, grbah, grbeam, grbf16, grbgd, grbic,
                      grblip, grbn, grbo, grbp, grbpe, grbptt, grbrnn, grca, grcae,
                      grcart, grcfm, grclp, grcos, grctr, grcvf, grcvs, grdae,
                      grdal, grdbs, grdcgan, grddim, grddqn, grdeit, grdetr)

A = {}
def put(k, v):
    A[k] = v

# ---- optimisers
r = hmadam.geron_adam([1.0, -2.0], m=[0.1, 0.0], v=[0.04, 0.0], eta=0.1, t=3)
put("adam", dict(m=r["m"].tolist(), v=r["v"].tolist(), mhat=r["m_hat"].tolist(),
                 vhat=r["v_hat"].tolist(), step=r["step"].tolist()))
r = gradmo.geron_adam_update([1.0], [0.1], [0.0], [0.0], t=1, eta=0.001)
put("adam_update", dict(theta=r["theta_new"], mhat=r["m_hat"], shat=r["s_hat"], step=r["step"]))
r = hmadmw.geron_adamw([1.0, 2.0], theta=[2.0, -1.0], eta=0.1, wd=0.5, t=2)
put("adamw", dict(theta=r["theta"].tolist(), adam=r["adam_step"].tolist(),
                  decay=r["decay_step"].tolist()))
r = hmadmx.geron_adamax([0.5], m=[0.1], u=[1.0], b1=0.9, b2=0.999, eta=0.1, t=2)
put("adamax", dict(u=r["u"].tolist(), step=r["step"].tolist(), mhat=r["m_hat"].tolist()))
r = hmadgr.geron_adagrad([2.0], s=[4.0], eta=0.1, eps=0.0)
put("adagrad", dict(s=r["s"].tolist(), step=r["step"].tolist(), eff=r["effective_lr"].tolist()))
r = grada2.geron_adagrad_update([1.0, 2.0], [2.0, -1.0], [0.0, 3.0], 0.1, eps=1e-10)
put("adagrad_update", dict(theta=r["theta_new"], s=r["s_new"], step=r["step"]))
r = gr1cy.geron_1cycle_schedule(0.1, 0.5, t=1, T=7)
put("cyc7", dict(lr=r["lr_schedule"], mom=r["momentum_schedule"], peak=r["peak_step"]))
r = gr1cy.geron_1cycle_schedule(0.1, 0.5, t=1, T=2)
put("cyc2", dict(lr=r["lr_schedule"], mom=r["momentum_schedule"], peak=r["peak_step"]))
put("bsz", [hmbsz.geron_batch_size_heuristic(1000)["batch_size"],
            hmbsz.geron_batch_size_heuristic(1000)["steps_per_epoch"],
            hmbsz.geron_batch_size_heuristic(50)["batch_size"],
            hmbsz.geron_batch_size_heuristic(10**6)["batch_size"],
            hmbsz.geron_batch_size_heuristic(10**6, memory_limit=100)["batch_size"]])

# ---- criteria / metrics
r = hmaic.geron_aic([-10.0, -9.0, -12.0], [2, 4, 1], n=20)
put("aic", dict(aic=r["aic"].tolist(), aicc=r["aicc"].tolist(),
                delta=r["delta"].tolist(), w=r["weights"].tolist(), best=r["best_index"]))
r = hmbic.geron_bic([-10.0, -8.0], [2, 6], 100)
put("bic", dict(bic=r["bic"].tolist(), w=r["weights"].tolist(), best=r["best_index"]))
put("gmm_np", [graic.gmm_n_params(3, 2, "full"), graic.gmm_n_params(3, 2, "diag"),
               graic.gmm_n_params(3, 2, "spherical"), graic.gmm_n_params(4, 5, "full")])
put("aic_gmm", graic.geron_aic_gmm(-100.0, 5)["aic"])
r = grbic.geron_bic_gmm(-100.0, 100, 5)
put("bic_gmm", dict(bic=r["bic"], per=r["penalty_per_param"], strict=r["stricter_than_aic"]))
yt = [0, 0, 1, 1, 0, 1]; sc = [0.1, 0.4, 0.35, 0.8, 0.4, 0.4]
r = hmauc.geron_auc_roc(yt, sc)
put("auc_mw", dict(auc=r["auc"], fpr=r["fpr"].tolist(), tpr=r["tpr"].tolist(),
                   thr=[x for x in r["thresholds"][1:]]))
r2 = grauc.geron_auc_roc(yt, sc)
put("auc_tz", dict(auc=r2["auc"], fpr=r2["fpr"], tpr=r2["tpr"]))
r = grcfm.geron_confusion_matrix([0, 0, 1, 1, 2, 2, 0], [0, 1, 1, 1, 2, 0, 2])
put("cfm", dict(m=r["matrix"], acc=r["accuracy"], prec=r["precision"],
                rec=r["recall"], f1=r["f1"], macro=r["macro_f1"], sup=r["support"]))
r = grcart.geron_cart_split_cost([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1], 0, 1.5)
put("cart_gini", dict(cost=r["cost"], l=r["impurity_left"], rr=r["impurity_right"],
                      par=r["impurity_parent"], dec=r["impurity_decrease"]))
r = grcart.geron_cart_split_cost([[1.0], [2.0], [3.0], [4.0]], [0, 0, 1, 1], 0, 1.5,
                                 criterion="entropy")
put("cart_ent", dict(cost=r["cost"], par=r["impurity_parent"]))
r = grcart.geron_cart_split_cost([[1.0], [2.0], [3.0], [4.0]], [1.0, 2.0, 8.0, 9.0], 0,
                                 2.5, criterion="mse")
put("cart_mse", dict(cost=r["cost"], par=r["impurity_parent"]))
r = hmbin.geron_binary_classification([[1.0, 2.0], [1.0, -3.0], [1.0, 0.0]],
                                      [0.5, 0.5], y_true=[1, 1, 0])
put("binclf", dict(p=r["p_hat"].tolist(), yp=r["y_pred"].tolist(), acc=r["accuracy"],
                   prec=r["precision"], rec=r["recall"], f1=r["f1"]))
r = hmbv.geron_bias_variance_tradeoff([[1.0, 2.0], [3.0, 4.0], [2.0, 1.0]], [2.0, 2.0])
put("bv", dict(b2=r["bias2"], v=r["variance"], noise=r["noise"], mse=r["mse"],
               mp=r["mean_pred"].tolist()))
r = hmbat.geron_batch_learning([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], [1.0, 3.0, 5.2],
                               ridge=0.5)
put("batlearn", dict(theta=r["theta"].tolist(), mse=r["train_mse"], r2=r["r2"]))
r = hmbgdg.geron_batch_gd_grad([[1.0, 1.0], [1.0, 2.0]], [1.0, 2.0], [0.0, 0.0], eta=0.1)
put("bgdg", dict(g=r["gradient"].tolist(), cost=r["cost"], tn=r["theta_next"].tolist()))
r = grbgd.geron_batch_gradient_descent([[1.0], [2.0]], [2.0, 4.0], [0.0], eta=0.1, n_iter=3)
put("bgd", dict(theta=r["theta"], loss=r["loss_history"], emax=r["eta_max_stable"]))
r = grcos.geron_conv_output_size([32, 28, 28], [5, 3, 3], padding=[2, 0, 1], stride=[1, 2, 1])
put("cos", dict(out=r["out_size"], rf=r["receptive_field"], sp=r["same_padding"],
                dr=r["dropped_cells"], same=r["is_same"]))
X33 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
r = grcvf.geron_conv2d_forward(X33, [[1.0, 0.0], [0.0, 1.0]], b=0.5, stride=1, padding=1)
put("cvf", dict(Y=r["Y"], shape=list(r["out_shape"]), ma=r["n_multiply_adds"]))

# ---- attention / rnn / nets
I2 = [[1.0, 0.0], [0.0, 1.0]]
r = hmbdn.geron_bahdanau_attention([[1.0, 0.0], [0.0, 1.0]], [0.5, -0.5], I2, I2, [1.0, 0.5],
                                   b=[0.1, -0.2])
put("bahd", dict(sc=r["scores"].tolist(), al=r["alpha"].tolist(), ctx=r["context"].tolist(),
                 ent=r["entropy"], am=r["argmax"]))
r = grbah.geron_bahdanau_attention([1.0], [[1.0], [3.0]], Wh=[[1.0]], Ws=[[2.0]], v=[1.0])
put("bahd_gr", dict(w=r["weights"], ctx=r["context"], sc=r["scores"], am=r["argmax"]))
r = grca.geron_cross_attention([[1.0, 2.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
                               WQ=[[1.0, 0.0], [0.0, 1.0]], WK=[[0.5, 0.0], [0.0, 1.0]],
                               WV=[[1.0], [2.0]])
put("crossattn", dict(out=r["output"], w=r["attention_weights"], sc=r["scale"]))
r = hmbrnn.geron_bidirectional_rnn([[1.0], [1.0], [0.5]], [[1.0]], [[0.5]], [[1.0]], [[-0.5]])
put("brnn", dict(out=r["output"].tolist(), f=r["h_fwd"].ravel().tolist(),
                 b=r["h_bwd"].ravel().tolist(), fin=r["final"].tolist()))
r = grbrnn.geron_bidirectional_rnn([[1.0], [2.0]], [[9.0], [8.0]],
                                   backward_in_reverse_order=True)
put("brnn_comb", r["h"])
r = hmbnm.geron_biological_neuron([[1.0, 2.0], [0.0, -1.0]], [0.5, -1.0], 0.25,
                                  activation="sigmoid")
put("neuron", dict(z=r["z"].tolist(), a=r["a"].tolist()))
r = hmbntr.geron_batch_normalization([[1.0, 4.0], [3.0, 0.0], [2.0, 2.0]], gamma=2.0,
                                     beta=5.0, eps=1e-5)
put("bn_hm", dict(y=r["y"].tolist(), xh=r["x_hat"].tolist(), mu=r["mu"].tolist(),
                  var=r["var"].tolist(), rm=r["running_mean"].tolist(),
                  rv=r["running_var"].tolist()))
r = grbn.geron_batch_normalization([[0.0, 1.0], [2.0, 5.0]], gamma=[3.0, 1.0],
                                   beta=[5.0, 0.0], eps=0.0, momentum=0.9)
put("bn_gr", dict(Y=r["Y"], xh=r["x_hat"], bm=r["batch_mean"], bv=r["batch_var"],
                  rm=r["running_mean"], rv=r["running_var"]))
W1 = [[1.0, -1.0]]; W2 = [[2.0], [3.0]]
r = hmbp.geron_backpropagation([[1.0], [2.0]], [[0.0], [1.0]], [W1, W2], ["relu", "identity"])
put("bp_mse", dict(loss=r["loss"], out=r["output"].tolist(),
                   g0=r["grads_W"][0].tolist(), g1=r["grads_W"][1].tolist(),
                   b0=r["grads_b"][0].tolist()))
r = hmbp.geron_backpropagation([[1.0, 0.0], [0.0, 1.0]], [0, 1],
                               [[[1.0, 0.5], [0.5, 1.0]]], ["softmax"], loss="ce")
put("bp_ce", dict(loss=r["loss"], out=r["output"].tolist(), g=r["grads_W"][0].tolist()))
r = grbp.geron_backpropagation_gradient([[[1.0, 2.0]], [[3.0]]], [[[1.0], [1.0]]], [[1.0]],
                                        activation="sigmoid")
put("bp_gr", dict(gw=r["grad_weights"][0], d=r["deltas"][-1], loss=r["loss"], gn=r["grad_norm"]))
r = grbptt.geron_backprop_through_time([[1.0, 0.5], [0.5, 1.0]], [[0.2, -0.1], [0.4, 0.3]],
                                       [[1.0, 0.0], [0.0, 2.0]],
                                       W_h=[[0.5, 0.1], [-0.2, 0.3]], h_init=[0.1, 0.2])
put("bptt", dict(wx=r["grad_Wx"], wh=r["grad_Wh"], b=r["grad_b"], d=r["deltas"],
                 nn=r["per_step_delta_norm"], ratio=r["vanishing_ratio"]))
r = graut.geron_autograd_chain_rule([[[2.0, 0.0], [0.0, 3.0]], [[1.0, 1.0]]], [1.0])
put("chain", dict(gi=r["grad_input"], inter=r["intermediate_grads"], gn=r["grad_norm"]))

# ---- contrastive / multimodal
r = grclp.geron_clip_contrastive_loss([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                                      [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], tau=0.5)
put("clip", dict(loss=r["loss"], i2t=r["loss_i2t"], t2i=r["loss_t2i"],
                 sim=r["similarity"], acc=r["accuracy_i2t"], ch=r["chance_loss"]))
r = grctr.geron_contrastive_infonce([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.2], [0.1, 1.0]],
                                    [[0.0, 1.0], [1.0, 1.0]], tau=0.5)
put("infonce", dict(loss=r["loss"], per=r["per_anchor_loss"], pos=r["pos_sim"],
                    neg=r["neg_sim"], hard=r["hardest_negative"], acc=r["accuracy"]))
r = hmblip.geron_blip([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                      [[1.0, 0.1], [0.2, 1.0], [1.0, 0.9]], temperature=0.7,
                      caption_logprobs=[-0.5, -1.0, -0.25])
put("blip", dict(itc=r["itc_loss"], i2t=r["itc_i2t"], t2i=r["itc_t2i"], itm=r["itm_loss"],
                 lm=r["lm_loss"], tot=r["total_loss"], sim=r["similarity"].tolist(),
                 ret=r["retrieval_acc"]))
cl = [[[0.0, 1.0], [2.0, 0.0]], [[1.0, 1.0], [0.0, 3.0]]]
r = grblip.geron_blip_itm_itc([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.2], [0.1, 1.0]],
                              cl, [[0, 1], [1, -1]], tau=0.5)
put("blip2loss", dict(loss=r["loss"], itc=r["itc"], itm=r["itm"], lm=r["lm"],
                      acc=r["itm_accuracy"], ppl=r["lm_perplexity"], sim=r["similarity"]))
r = hmblp2.geron_blip2([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [1.0, 0.0], n_query=3,
                       d_query=4, d_llm=2, seed=5)
put("blip2", dict(q=r["query_output"].tolist(), llm=r["llm_input"].tolist(),
                  att=r["attention"].tolist(), sim=r["similarity"],
                  qs=r["query_similarities"].tolist(), tp=r["trainable_params"]))
r = grdeit.geron_deit_distillation_loss([[1.0, 0.0], [0.0, 2.0]], [[0.5, 0.5], [1.0, 0.0]],
                                        [0, 1], [[1.0, 0.0], [1.0, 0.0]], alpha=0.3)
put("deit", dict(loss=r["loss"], lc=r["loss_cls"], ld=r["loss_dist"], tl=r["teacher_labels"],
                 ta=r["teacher_agreement"], ac=r["accuracy_cls"], ad=r["accuracy_dist"]))
skew = lambda ctx: np.array([0.0, 1.0, 2.0]) + 0.1 * len(ctx)
r = grdal.geron_dalle_autoregressive_token([0, 1], [2, 0], skew, temperature=0.5)
put("dalle", dict(ll=r["log_likelihood"], tl=r["token_logprobs"], ppl=r["perplexity"],
                  ntp=r["next_token_probs"], nt=r["next_token"]))
r = grdal.geron_dalle_autoregressive_token([0], [], lambda c: np.array([0.0, 5.0]), top_k=1)
put("dalle_topk", r["next_token_probs"])
pb = [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0], [0.0, 0.0, 2.0, 2.0]]
pc = [[10.0, 0.0], [0.0, 10.0], [1.0, 1.0]]
r = grdetr.geron_detr_hungarian_matching(pb, pc, [[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0]],
                                         [0, 1], no_object_class=1)
put("detr", dict(m=[list(t) for t in r["matching"]], cost=r["cost_matrix"],
                 tc=r["total_cost"], loss=r["loss"], lc=r["loss_class"], lb=r["loss_bbox"],
                 lg=r["loss_giou"], ln=r["loss_no_object"], gi=r["matched_giou"],
                 l1=r["matched_l1"], un=r["unmatched_predictions"]))

# ---- RL / diffusion
r = grac.geron_actor_critic_advantage([0.0, 1.0, 2.0], [0, 1, 2], [1, 2, 0],
                                      [1.0, -1.0, 0.5], 0.9, done=[False, True, False])
put("ac", dict(a=r["advantage"], t=r["td_target"], cl=r["critic_loss"]))
r = grddqn.geron_double_dqn_target([[0.0, 1.0], [2.0, 0.0]], [[10.0, -10.0], [1.0, 3.0]],
                                   s_next=[0, 1, 0], r=[0.0, 1.0, 2.0], gamma=0.9,
                                   done=[False, False, True])
put("ddqn", dict(t=r["target"], a=r["selected_action"], v=r["vanilla_target"],
                 g=r["overestimation_gap"]))
P = [[[0.7, 0.3], [0.2, 0.8]], [[0.5, 0.5], [0.1, 0.9]]]
R = [[1.0, 0.0], [0.0, 2.0]]
r = hmbel.geron_bellman_optimality([0.0, 0.0], P, R, 0.9)
put("vi", dict(V=r["V"].tolist(), pol=r["policy"].tolist(), Q=r["Q"].tolist(),
               it=r["iterations"], res=r["residual"]))
T = [[[0.7, 0.3], [0.2, 0.8]], [[0.5, 0.5], [0.1, 0.9]]]
r = grbo.geron_bellman_optimality([[0.0, 0.0], [0.0, 0.0]], T, [[1.0, 0.0], [0.0, 2.0]], 0.9)
put("qvi", dict(Q=r["Q"], V=r["V"], pol=r["policy"], it=r["iterations"]))
ab = [1.0, 0.9, 0.8, 0.5]
r = grddim.geron_ddim_sampling_step([1.0, -0.5], t=3, t_prev=1, eps_pred=[0.2, 0.1],
                                    alpha_bar=ab)
put("ddim", dict(xp=r["x_prev"], x0=r["x0_pred"], sig=r["signal_scale"], noi=r["noise_scale"]))
env = {"reset": lambda: [1.0], "step": lambda a: ([1.0], 1.0 if a == 0 else 0.0, True)}
r = hma2c.geron_a2c(env, [[0.0], [0.0]], [0.0], epochs=25, lr=0.5, seed=1)
put("a2c", dict(actor=r["actor"].tolist(), critic=r["critic"].tolist(),
                ret=r["returns"].tolist(), pol=r["policy"]([1.0]).tolist(),
                val=r["value"]([1.0])))
r = hma3c.geron_a3c(env, [[0.0], [0.0]], [0.0], n_workers=3, epochs=10, lr=0.5, seed=3)
put("a3c", dict(actor=r["actor"].tolist(), critic=r["critic"].tolist(),
                wr=r["worker_returns"].tolist(), upd=r["updates"],
                pol=r["policy"]([1.0]).tolist()))

# ---- clustering / trees / ensembles
r = grdbs.geron_dbscan_core_point([[0.0], [0.5], [1.2], [10.0]], eps=1.0, min_samples=2)
put("dbscan", dict(c=r["is_core"], b=r["is_border"], nz=r["is_noise"],
                   cnt=r["neighbor_counts"], nb=r["neighbors"]))
r = hmagc.geron_agglomerative([[0.0], [1.0], [10.0], [11.0], [5.0]], 2, linkage="average")
put("agg", dict(lab=r["labels"].tolist(), m=[list(t) for t in r["merges"]],
                h=[float(x) for x in r["heights"]]))
r = hmbrch.geron_birch([[0.0], [0.1], [0.2], [10.0], [10.1]], n_clusters=2,
                       threshold=0.5, branching_factor=2)
put("birch", dict(lab=r["labels"].tolist(), c=r["subcluster_centers"].ravel().tolist(),
                  sl=r["subcluster_labels"].tolist(), sz=r["subcluster_sizes"].tolist(),
                  ns=r["n_subclusters"], rad=r["radii"].tolist(), nl=r["n_leaves"]))
r = hmadab.geron_adaboost([[1.0], [2.0], [3.0], [4.0], [5.0]], [1, -1, 1, -1, 1],
                          n_estimators=3)
put("adaboost", dict(al=r["alphas"].tolist(), er=r["errors"].tolist(),
                     te=r["train_errors"].tolist(), dec=r["decision"].tolist(),
                     w=r["weights"].tolist(),
                     pred=[float(v) for v in r["predict"]([[1.0], [4.0]])]))
r = gradaw.geron_adaboost_weight_update([0, 0, 1, 1], [0, 1, 1, 1], [0.25] * 4, math.log(3.0))
put("adawgt", dict(w=r["weights_new"], e=r["weighted_error"], bf=r["boost_factor"]))
r = hmbag.geron_bagging([[1.0], [2.0], [3.0], [4.0], [5.0]], [1.0, 1.0, 5.0, 5.0, 6.0],
                        n_estimators=6, seed=1)
put("bagging", dict(tp=r["train_pred"].tolist(), mse=r["train_mse"],
                    oob=[None if np.isnan(v) else v for v in r["oob_pred"].tolist()],
                    om=r["oob_mse"], cov=r["oob_coverage"],
                    pr=r["predict"]([[1.0], [5.0]]).tolist()))
r = grbag.geron_bagging_predictor([[1.0, 2.0], [3.0, 4.0], [2.0, 9.0]])
put("bagpred", dict(p=r["prediction"], v=r["per_instance_variance"],
                    md=r["mean_disagreement"], se=r["se"]))
r = grbag.geron_bagging_predictor([[0, 1, 1], [1, 1, 0], [1, 0, 0]], aggregate="vote")
put("bagvote", r["prediction"])
r = grcvs.geron_cross_validation_score([[1.0], [2.0], [3.0], [4.0], [5.0]],
                                       [2.0, 4.0, 6.0, 8.0, 10.5], K=2)
put("cv", dict(cv=r["cv_score"], fs=r["fold_scores"], sz=r["fold_sizes"], se=r["se"],
               worst=r["worst_fold"], spread=r["spread"]))

# ---- autoencoders
r = hmaen.geron_autoencoder([[0.0, 0.0], [1.0, 1.1], [2.0, 2.3], [3.0, 2.9]], 1)
put("linae", dict(err=r["recon_error"], evr=r["explained_variance_ratio"].tolist(),
                  recon=r["reconstruction"].tolist(), mean=r["mean"].tolist()))
r = grael.geron_autoencoder_reconstruction_loss([[1.0, 2.0], [3.0, 1.0]], [[0.5], [0.2]],
                                                [[1.0, 3.0], [2.0, 1.0]])
put("aeloss", dict(loss=r["loss"], mse=r["mse_per_element"], ps=r["per_sample_loss"],
                   cr=r["compression_ratio"], ev=r["explained_variance"]))
r = grdae.geron_denoising_autoencoder([[1.0, 2.0], [0.0, 1.0]], [[0.1, -0.1], [0.2, 0.2]],
                                      [[1.0, 3.0], [0.1, 1.0]])
put("dae", dict(loss=r["loss"], xt=r["x_tilde"], ne=r["noise_energy"],
                dg=r["denoising_gain"], snr=r["snr_db"], mse=r["mse_per_element"]))
r = grcae.geron_convolutional_autoencoder([[1.0, 2.0], [3.0, 4.0]], [[[1.0]]],
                                          [[[1.0, 1.0], [1.0, 1.0]]])
put("cae", dict(code=r["code"], xh=r["x_hat"], loss=r["loss"], cr=r["compression_ratio"]))
zero = lambda A: np.zeros_like(np.asarray(A, dtype=float))
r = hmanae.geron_anomaly_autoencoder(zero, [[0.0], [3.0], [1.0], [5.0]], quantile=0.6)
put("anae", dict(e=r["errors"].tolist(), f=[bool(b) for b in r["is_anomaly"]],
                 thr=r["threshold"], na=r["n_anomalies"]))
W0 = [[1.0, 1.0, 1.0, 1.0], [0.5, -0.5, 0.25, 1.0]]
K1 = [[1.0, 0.5], [0.5, 1.0]]
r = grdcgan.geron_dcgan_generator([1.0, 2.0], [W0, K1], seed_shape=(2, 2))
put("dcgan", dict(img=r["image"], shape=list(r["image_shape"]), seed=r["seed"],
                  uf=r["upsample_factor"]))
Xa = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]; ya = [1.0, 2.0, 3.0, 4.0]
Xt = [[1.0, 1.0], [2.0, 0.0]]; yt2 = [3.0, 2.0]
r = hmauxpt.geron_auxiliary_task_pretraining(None, (Xa, ya), (Xt, yt2), aux_epochs=50, epochs=5)
put("auxpt", dict(tp=r["theta_pretrained"].tolist(), th=r["theta"].tolist(),
                  ts=r["theta_scratch"].tolist(), tl=r["target_loss"],
                  sl=r["scratch_loss"], tg=r["transfer_gain"],
                  al=r["aux_losses"].tolist()[:3], fl=r["finetune_losses"].tolist()))
ident = lambda A: np.asarray(A, dtype=float)
r = hmbftn.geron_bert_finetune(ident, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0, 1, 0],
                               epochs=25, lr=0.5, l2=0.01)
put("bftn", dict(W=r["W"].tolist(), b=r["b"].tolist(), l=r["losses"].tolist()[:3],
                 lend=r["losses"].tolist()[-1], acc=r["accuracy"],
                 p=r["probabilities"].tolist()))

# ---- bf16
vals = [1.0, 1.1, -1.1, 1.00390625, 0.0, 1.0078125, 3.0e38, 1e-40]
r = hmbf16.geron_bf16(vals)
put("bf16", dict(v=[float(x) for x in r["values"]], bits=r["bits"],
                 ae=r["abs_error"].tolist(), mre=r["max_rel_error"]))
put("bf16_trunc", [float(x) for x in hmbf16.geron_bf16([1.1, -1.1, 1.9],
                                                       rounding="truncate")["values"]])
r = grbf16.geron_bf16_range([1.0, 1.0078125, -2.0, 1.00390625, 1e-40, 3.0e38])
put("bf16r", dict(b=r["bf16"], mre=r["max_rel_error"], eps=r["machine_eps"],
                  ov=r["n_overflow"], un=r["n_underflow"], ex=r["exact"]))

# ---- autograd tape
f = lambda p: p[0] * p[1] + p[0].exp()
r = hmagrd.geron_autograd(f, [0.0, 3.0])
put("autograd1", dict(v=r["value"], g=r["grad"].tolist(), ts=r["tape_size"]))
g2 = lambda p: ((p[0] ** 2 + 1.0).log() * p[1].tanh() - p[0] / p[1]).sigmoid()
r = hmagrd.geron_autograd(g2, [2.0, 1.5])
put("autograd2", dict(v=r["value"], g=r["grad"].tolist(), ts=r["tape_size"]))

# ---- time series / decoding
y = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
r = hmarim.geron_arima(y, p=1, d=0, q=0, include_mean=False)
put("arima_ar", dict(ar=r["ar"].tolist(), s2=r["sigma2"], fc=r["forecast"](2).tolist()))
r = hmarim.geron_arima([1.0, 3.0, 5.0, 7.0], p=0, d=1, q=0)
put("arima_d1", dict(ic=r["intercept"], fc=r["forecast"](2).tolist(), s2=r["sigma2"]))
y2 = [1.0, 2.0, 1.5, 2.5, 2.0, 3.0, 2.5, 3.5, 3.0, 4.0, 3.5, 4.5]
r = hmarim.geron_arima(y2, p=1, d=0, q=1)
put("arima_arma", dict(ar=r["ar"].tolist(), ma=r["ma"].tolist(), ic=r["intercept"],
                       s2=r["sigma2"], aic=r["aic"], fc=r["forecast"](2).tolist()))
r = grarma.geron_arima_forecast([1.0, 2.0, 3.0, 4.5], phi=[0.5], theta=[0.25], d=1)
put("armafc", dict(f=r["forecast"], fd=r["forecast_differenced"], e=r["residuals"],
                   w=r["differenced"], s2=r["sigma2"]))
lp = np.log([0.5, 0.3, 0.2])
r = hmbms.geron_beam_search(lambda s, prefix: lp, None, beam_width=2, max_len=3)
put("beam", dict(seq=[int(t) for t in r["sequence"]], sc=r["score"],
                 beams=[list(map(int, b)) for b in r["beams"]], scores=r["scores"],
                 fin=r["finished"]))
r = hmbms.geron_beam_search(lambda s, prefix: np.log([0.1, 0.9]), None, beam_width=2,
                            max_len=3, eos=1)
put("beam_eos", dict(seq=[int(t) for t in r["sequence"]], sc=r["score"]))
S = [[-0.1, -2.0, -3.0], [-0.2, -3.0, -0.15], [-1.0, -0.5, -2.0]]
r = grbeam.geron_beam_search_decoder(S, beam_width=3, length_penalty=0.5)
put("beamdec", dict(bs=r["best_sequence"], sc=r["best_score"],
                    beams=[[list(map(int, s)), v] for s, v in r["beams"]],
                    ns=r["normalised_scores"], gr=r["greedy_sequence"]))
corpus = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
r = hmbpet.geron_bpe_tokenizer(corpus, vocab_size=20)
put("bpe_hm", dict(m=[list(t) for t in r["merges"]], nm=r["n_merges"],
                   tok=r["tokenize"]("newest"), tok2=r["tokenize"]("lowest"),
                   vocab=list(r["vocab"].keys())))
put("bpe_hm13", hmbpet.geron_bpe_tokenizer(corpus, vocab_size=13)["n_merges"])
r = grbpe.geron_bpe_tokenizer_merge({"low": 5, "lowest": 2, "newer": 3}, 4)
put("bpe_gr", dict(m=[list(t) for t in r["merges"]], c=r["merge_counts"],
                   v=r["vocab"], s={k: v for k, v in r["splits"].items()},
                   nb=r["n_tokens_before"], na=r["n_tokens_after"], comp=r["compression"]))

# ---- transformers
ids = [1, 2, 3, 4, 5, 6, 0, 2]
r = hmalbt.geron_albert(ids, n_layers=3, n_heads=2, d_model=8, d_embed=4, vocab_size=7, seed=2)
put("albert", dict(h=r["hidden"][0].tolist(), np_=r["n_params"], nu=r["n_params_unshared"],
                   bp=r["block_params"], ep=r["embedding_params"], ed=r["embedding_params_direct"]))
r = hmbert.geron_bert(ids, n_layers=2, n_heads=2, d_model=8, vocab_size=7, seed=1)
put("bert", dict(h=r["hidden"][0].tolist(), loss=r["mlm_loss"], mp=r["masked_positions"][0],
                 nsp=r["nsp_logits"].tolist(), npar=r["n_params"],
                 a0=r["attentions"][0][0][0].tolist()))
r = hmbrob.geron_roberta(ids, n_layers=2, n_heads=2, d_model=8, vocab_size=7, epochs=3, seed=1)
put("roberta", dict(h=r["hidden"][0].tolist(), el=r["epoch_losses"].tolist(),
                    masks=r["masks"], npar=r["n_params"], nw=r["n_params_with_nsp"]))
src = ["the", "cat", "sat", "on", "the", "mat", "today", "ok"]
tgt = ["the", "cat", "sat", "on", "the", "mat"]
r = hmbart.geron_bart(src, tgt, mask_ratio=0.35, mean_span=2.0, seed=2)
put("bart", dict(cor=r["corrupted"], sp=[list(map(int, s)) for s in r["spans"]],
                 nm=r["n_masked"], loss=r["loss"], ppl=r["perplexity"],
                 tlp=r["token_logprobs"].tolist()))
r = hmalex.geron_alexnet(1000)
put("alexnet", dict(tp=r["total_params"], fd=r["flatten_dim"],
                    conv=[l["out"] for l in r["layers"] if l["kind"] == "conv"],
                    cp=r["conv_params"], fp=r["fc_params"]))
put("alexnet10", hmalex.geron_alexnet(10)["total_params"])
r = hmalex.geron_alexnet(5, input_size=128, in_channels=1)
put("alexnet128", dict(tp=r["total_params"], fd=r["flatten_dim"]))

print(json.dumps(A))
