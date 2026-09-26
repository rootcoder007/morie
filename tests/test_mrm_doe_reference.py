"""mrm_doe ANOVA tables against R's anova(lm()) (sequential sums of squares)."""


def test_rcbd_unbalanced_matches_r_anova():
    # reference: anova(lm(y ~ block + trt)) with cell (b3, t1) removed
    import math

    import pandas as pd

    from morie import mrm_doe as M

    rows = []
    for k in range(20):
        b, t = k % 5 + 1, k // 5 + 1
        rows.append({"block": f"b{b}", "trt": f"t{t}", "y": math.sin((k + 1) * 1.3) + t * 0.4 + b * 0.2})
    del rows[2]
    tab = M.mrm_rcbd(pd.DataFrame(rows), response_col="y", treatment_col="trt", block_col="block")["anova"]
    p = list(tab["p_value"])
    assert abs(p[0] - 2.7041850005459e-06) < 1e-15
    assert abs(p[1] - 1.90660263699952e-05) < 1e-14
    assert list(tab["df"]) == [4, 3, 11]
