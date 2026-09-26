"""morie.validation discrimination against pROC."""

from morie import validation as V


def test_auc_interval_is_delong():
    # pROC::ci.auc(roc(y, p), method = "delong")
    y = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1]
    p = [0.8, 0.3, 0.6, 0.9, 0.4, 0.2, 0.55, 0.6, 0.7, 0.1, 0.35, 0.45, 0.8, 0.5, 0.65, 0.25, 0.3, 0.15, 0.75, 0.4]
    r = V.assess_discrimination(y, p)
    assert abs(r.auroc - 0.94) <= 1e-14
    assert abs(r.auroc_ci_lower - 0.84577665807570845) <= 1e-12
    assert r.auroc_ci_upper == 1.0
