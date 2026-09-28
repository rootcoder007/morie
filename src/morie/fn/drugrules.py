# morie.fn -- function file (rootcoder007/morie)
"""Rule-based oral drug-likeness filters from computed molecular descriptors: Lipinski's rule of
five, Veber's rotatable-bond and polar-surface-area rules, the Egan egg and their conjunction as an
oral-bioavailability screen. Descriptors (molecular weight, Crippen logP, H-bond donors and
acceptors, rotatable bonds, topological polar surface area) are supplied by the caller."""

from __future__ import annotations

from ._richresult import RichResult

__all__ = ["lipinski_rule_of_five", "veber_rules", "egan_egg", "oral_bioavailability_rules"]


def lipinski_rule_of_five(mw, logp, hbd, hba, max_violations=1):
    r"""Lipinski's rule of five: MW <= 500, logP <= 5, H-bond donors <= 5, acceptors <= 10.

    A compound passes when it violates at most ``max_violations`` (default 1,
    i.e. at least three criteria met) of the four limits.

    References
    ----------
    Lipinski, C. A., Lombardo, F., Dominy, B. W. and Feeney, P. J. (1997).
    Experimental and computational approaches to estimate solubility and
    permeability in drug discovery and development settings. *Advanced Drug
    Delivery Reviews* 23, 3-25.

    Examples
    --------
    >>> r = lipinski_rule_of_five(520.0, 5.6, 2, 7)
    >>> r.violations, r.passes
    (2, False)
    """
    crit = {"mw": mw <= 500, "logp": logp <= 5, "hbd": hbd <= 5, "hba": hba <= 10}
    v = sum(1 for ok in crit.values() if not ok)
    return RichResult(payload={"violations": v, "passes": v <= max_violations, "criteria": crit})


def veber_rules(rotatable_bonds, psa=None, hbond_total=None):
    r"""Veber et al. (2002) oral bioavailability rules: rotatable bonds <= 10 and PSA <= 140 A^2.

    The polar-surface-area criterion may be replaced by the equivalent total
    H-bond count (donors + acceptors) <= 12.

    References
    ----------
    Veber, D. F., Johnson, S. R., Cheng, H.-Y., Smith, B. R., Ward, K. W. and
    Kopple, K. D. (2002). Molecular properties that influence the oral
    bioavailability of drug candidates. *Journal of Medicinal Chemistry* 45,
    2615-2623.

    Examples
    --------
    >>> veber_rules(8, psa=95.0).passes
    True
    """
    if psa is None and hbond_total is None:
        raise ValueError("give psa or hbond_total")
    polar = psa <= 140 if psa is not None else hbond_total <= 12
    ok = rotatable_bonds <= 10 and polar
    return RichResult(payload={"passes": ok, "rotatable_ok": rotatable_bonds <= 10, "polar_ok": polar})


def egan_egg(psa, logp):
    r"""Egan et al. (2000) absorption egg: PSA <= 131.6 A^2 and -1 <= AlogP98 <= 5.88.

    The 99 percent confidence ellipse of well-absorbed compounds in the
    (PSA, logP) plane is summarised by these limits; compounds inside are
    predicted to be well absorbed.

    References
    ----------
    Egan, W. J., Merz, K. M. and Baldwin, J. J. (2000). Prediction of drug
    absorption using multivariate statistics. *Journal of Medicinal
    Chemistry* 43, 3867-3877.

    Examples
    --------
    >>> egan_egg(90.0, 6.1).passes
    False
    """
    ok = psa <= 131.6 and -1.0 <= logp <= 5.88
    return RichResult(payload={"passes": ok, "psa_ok": psa <= 131.6, "logp_ok": -1.0 <= logp <= 5.88})


def oral_bioavailability_rules(mw, logp, hbd, hba, rotatable_bonds, psa):
    r"""Composite oral-bioavailability screen: Lipinski, Veber and Egan filters together.

    Returns each filter and the number passed; ``passes`` requires all three.

    Examples
    --------
    >>> r = oral_bioavailability_rules(350.0, 2.5, 2, 5, 6, 80.0)
    >>> r.n_passed, r.passes
    (3, True)
    """
    lip = lipinski_rule_of_five(mw, logp, hbd, hba).passes
    veb = veber_rules(rotatable_bonds, psa).passes
    egg = egan_egg(psa, logp).passes
    n = int(lip) + int(veb) + int(egg)
    return RichResult(payload={"lipinski": lip, "veber": veb, "egan": egg, "n_passed": n, "passes": n == 3})


def cheatsheet() -> str:
    return "lipinski_rule_of_five / veber_rules / egan_egg / oral_bioavailability_rules -> drug-likeness filters."
