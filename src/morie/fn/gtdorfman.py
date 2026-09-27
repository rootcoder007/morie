"""Operating characteristics of two-stage (Dorfman) group testing, heterogeneous risks allowed.

Bilder & Loughin (2025), Analysis of Categorical Data with R, eqs (6.26)-(6.29).
"""

from ._richresult import RichResult

__all__ = ["gtdorfman"]


def gtdorfman(p, se, sp, se_r=None, sp_r=None, size=None):
    """Expected tests and pooling accuracy for one group tested by Dorfman's algorithm.

    A group of I specimens (probabilities p_i) is tested once; if positive, each
    specimen is retested. With P(Z = 1) = Se (1 - prod(1 - p)) + (1 - Sp) prod(1 - p),
    E(T) = 1 + I P(Z = 1) (6.26, 6.28; E(T) = 1 when I = 1). The pooling
    sensitivity is PSe = Se Se_R; the probability that specimen i is finally
    called positive is
    P(Y_i = 1) = (1 - Sp)(1 - Sp_R) prod_all(1 - p) + Se Se_R p_i
    + Se (1 - Sp_R)(1 - p_i)(1 - prod_{j != i}(1 - p_j)), giving
    PSp_i = 1 - (P(Y_i = 1) - PSe p_i)/(1 - p_i) (6.27/6.29),
    PPPV_i = PSe p_i / P(Y_i = 1) and PNPV_i = PSp_i (1 - p_i)/(1 - P(Y_i = 1)).
    Overall values weight the specimens as binGroup2 does.

    Parameters
    ----------
    p : float or sequence of floats
        Common probability (then give ``size``) or one probability per specimen.
    se, sp : float
        Accuracy of the group test.
    se_r, sp_r : float, optional
        Accuracy of the retests (default: se, sp).
    size : int, optional
        Group size when p is a single number.

    Returns
    -------
    RichResult
        Keys: expected_tests, tests_per_specimen, p_group_positive, individual (list
        of dicts PSe, PSp, PPPV, PNPV), overall (dict).

    References
    ----------
    Dorfman, R. (1943). Annals of Mathematical Statistics 14, 436-440.
    Bilder, C. R. & Loughin, T. M. (2025). Analysis of Categorical Data with R
    (2nd ed.). CRC Press. Eqs (6.26)-(6.29).

    Examples
    --------
    >>> round(gtdorfman(0.05, 0.95, 0.97, 0.9, 0.99, size=6)["expected_tests"], 11)
    2.64229276375
    """
    ps = [float(p)] * int(size) if size is not None else [float(v) for v in p]
    se_r = se if se_r is None else se_r
    sp_r = sp if sp_r is None else sp_r
    size_i = len(ps)
    if size_i < 1 or any(not 0 <= v < 1 for v in ps):
        raise ValueError("need at least one specimen and 0 <= p < 1")
    allneg = 1.0
    for v in ps:
        allneg *= 1 - v
    pz = se * (1 - allneg) + (1 - sp) * allneg
    et = 1.0 if size_i == 1 else 1 + size_i * pz
    pse = se * se_r if size_i > 1 else se
    ind = []
    for pi in ps:
        if size_i == 1:
            py = se * pi + (1 - sp) * (1 - pi)
            psp = sp
        else:
            others = allneg / (1 - pi)
            py = (1 - sp) * (1 - sp_r) * allneg + se * se_r * pi + se * (1 - sp_r) * (1 - pi) * (1 - others)
            psp = 1 - (py - pse * pi) / (1 - pi)
        ind.append({"PSe": pse, "PSp": psp, "PPPV": pse * pi / py, "PNPV": psp * (1 - pi) / (1 - py), "P(Y=1)": py})
    sp_ = sum(ps)
    overall = {
        "PSe": sum(pi * d["PSe"] for pi, d in zip(ps, ind)) / sp_ if sp_ > 0 else pse,
        "PSp": sum((1 - pi) * d["PSp"] for pi, d in zip(ps, ind)) / sum(1 - v for v in ps),
        "PPPV": sum(pi * d["PSe"] for pi, d in zip(ps, ind)) / sum(d["P(Y=1)"] for d in ind),
        "PNPV": sum((1 - pi) * d["PSp"] for pi, d in zip(ps, ind)) / sum(1 - d["P(Y=1)"] for d in ind),
    }
    return RichResult(
        title="Dorfman group testing",
        summary_lines=[("E(T)", et), ("tests per specimen", et / size_i)],
        payload={
            "expected_tests": et,
            "tests_per_specimen": et / size_i,
            "p_group_positive": pz,
            "individual": ind,
            "overall": overall,
        },
    )


def cheatsheet():
    return (
        "gtdorfman: Dorfman expected tests, PSe, PSp, PPPV, PNPV (heterogeneous p). Bilder & Loughin eqs (6.26)-(6.29)."
    )
