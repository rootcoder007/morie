"""Cytochrome P450 inhibition: descriptors for the five isozymes that
matter, and a model you have to fit before it will tell you anything.

Most drugs are cleared by five cytochrome P450 isozymes -- 1A2, 2C9,
2C19, 2D6 and 3A4 -- and a compound that inhibits one of them will
change the blood level of every other drug that isozyme clears. That is
where drug-drug interactions come from, and it is why an inhibition
profile is run early. Veith and colleagues measured it directly:
seventeen thousand compounds against all five isozymes in quantitative
high-throughput screen, which is a DATASET, and then a model fitted to
that dataset.

The distinction matters here, because the two halves are not equally
available. The assay's design and its findings are published. The
fitted model's coefficients are not published as a table anyone can
copy, and this module will not invent them.

So it is built the way the paper is built, in two pieces:

  THE DESCRIPTORS. Computed exactly from the molecular graph, with
  nothing fitted and nothing approximated: molecular weight from the
  IUPAC standard atomic weights, heavy-atom count, rings and aromatic
  rings, rotatable bonds by their definition, hydrogen-bond donors and
  acceptors and basic nitrogens from the pharmacophore typing, the
  halogen count, the polar-atom count, and the fraction of carbons that
  are saturated. Every one of these is checkable by hand and several
  are checked by hand in the anchors.

  THE MODEL. A logistic regression, fitted by iteratively reweighted
  least squares, that the CALLER trains on their own inhibition data.
  ``fit`` returns coefficients; ``cyp450_inhibition`` applies them.

Called with no model, the function returns the descriptors, a
``predicted`` of None, and a ``reason`` saying in words that no
coefficients were supplied and that the paper does not publish a set to
default to. A refusal that names what is missing is worth more than a
number nobody can trace, and it is anchored: the module must refuse,
and must stop refusing the moment it is given a model.

What the paper does establish, and what the descriptors are chosen to
carry, is that inhibition rises with lipophilicity and with aromatic
ring count, that 2C19 and 3A4 are the most promiscuous of the five, and
that 2D6 responds to a basic nitrogen. Those are directions, not
coefficients, and they are stated here as directions.

References
  Veith, H., Southall, N., Huang, R., James, T., Fayne, D., Artemenko,
    N., Shen, M., Inglese, J., Austin, C.P., Lloyd, D.G. and Auld, D.S.
    (2009) "Comprehensive characterization of cytochrome P450 isozyme
    selectivity across chemical libraries." Nature Biotechnology 27(11),
    1050-1055. doi:10.1038/nbt.1581. The assay, the five isozymes and
    the selectivity findings.
  Meija, J., Coplen, T.B., Berglund, M., Brand, W.A., De Bievre, P.,
    Groning, M., Holden, N.E., Irrgeher, J., Loss, R.D., Walczyk, T.
    and Prohaska, T. (2016) "Atomic weights of the elements 2013."
    Pure and Applied Chemistry 88(3), 265-291. The masses used below.
  McCullagh, P. and Nelder, J.A. (1989) "Generalized Linear Models",
    2nd edition, Chapman and Hall. The iteratively reweighted least
    squares the fit uses.
"""

import math

from . import _w3num as _w
from .avalon import parse_smiles, _adjacency, implicit_h, ring_bonds
from .scfhop import atom_types
from ._richresult import RichResult

__all__ = ["cyp450_inhibition", "descriptors", "fit", "predict",
           "ISOZYMES", "cheatsheet"]

# The five isozymes Veith et al. screened, in the order the paper lists
# them.
ISOZYMES = ("1A2", "2C9", "2C19", "2D6", "3A4")

# IUPAC standard atomic weights. Where the element has a published
# interval rather than a single value, the conventional value is used
# and that is a documented choice, not a rounding.
_MASS = {"H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007,
         "O": 15.999, "F": 18.998403163, "P": 30.973761998,
         "S": 32.06, "Cl": 35.45, "Br": 79.904, "I": 126.90447}

# The descriptor block, in a fixed order, because a coefficient vector
# is meaningless without knowing which slot is which.
NAMES = ("mw", "heavy_atoms", "n_rings", "n_aromatic_rings",
         "n_rotatable", "hbd", "hba", "n_basic_n", "n_halogen",
         "n_polar", "fsp3", "formal_charge")


def descriptors(smiles):
    """The descriptor block, computed exactly from the graph.

    Rotatable bonds are single bonds outside a ring whose two ends each
    have more than one heavy neighbour -- the standard definition, which
    excludes a terminal methyl because spinning it changes nothing, and
    excludes ring bonds because they cannot turn.

    The saturated fraction counts carbons with four single bonds and no
    aromaticity, over all carbons; a molecule with no carbon has no
    such fraction and reports zero rather than dividing by nothing.
    """
    el, arom, chg, hexp, bonds, closures = parse_smiles(smiles)
    n = len(el)
    adj = _adjacency(n, bonds)
    nh = implicit_h(el, arom, chg, hexp, bonds)
    rings, inring = ring_bonds(n, bonds, closures)
    ty = atom_types(smiles)

    mass = []
    for i in range(n):
        m = _MASS.get(el[i])
        if m is None:
            raise ValueError("no standard atomic weight for " + el[i])
        mass.append(m + nh[i] * _MASS["H"])
    mw = _w.csum(mass)

    ringbond = {}
    for r in rings:
        for k in range(len(r)):
            a = r[k]
            b = r[(k + 1) % len(r)]
            ringbond[(a, b) if a < b else (b, a)] = True
    rot = 0
    for a, b, o in bonds:
        if o != 1:
            continue
        key = (a, b) if a < b else (b, a)
        if key in ringbond:
            continue
        if len(adj[a]) > 1 and len(adj[b]) > 1:
            rot += 1

    naro = 0
    for r in rings:
        allaro = True
        for v in r:
            if not arom[v]:
                allaro = False
        if allaro:
            naro += 1

    hbd = 0
    hba = 0
    basic = 0
    for i in range(n):
        if "D" in ty[i]:
            hbd += 1
        if "A" in ty[i]:
            hba += 1
        if "P" in ty[i]:
            basic += 1
    hal = 0
    pol = 0
    for i in range(n):
        if el[i] in ("F", "Cl", "Br", "I"):
            hal += 1
        if el[i] in ("N", "O"):
            pol += 1

    ncarb = 0
    nsp3 = 0
    for i in range(n):
        if el[i] != "C":
            continue
        ncarb += 1
        if arom[i]:
            continue
        allsingle = True
        for v, o, k in adj[i]:
            if o != 1:
                allsingle = False
        if allsingle and len(adj[i]) + nh[i] == 4:
            nsp3 += 1

    return [mw, float(n), float(len(rings)), float(naro), float(rot),
            float(hbd), float(hba), float(basic), float(hal),
            float(pol),
            (nsp3 / float(ncarb)) if ncarb else 0.0,
            float(sum(chg))]


def _logistic(z):
    """The logistic function, written so it cannot overflow either way.

    A large positive z would overflow exp(-z) in the naive form and a
    large negative one would overflow exp(z); branching on the sign
    keeps both exponentials below one.
    """
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit(X, y, ridge=1e-6, iters=50, tol=1e-12):
    """Logistic regression by iteratively reweighted least squares.

    An intercept is prepended, so the returned vector is one longer
    than a descriptor row and its first entry is the intercept.

    ``ridge`` is a small quadratic penalty on the slopes, not on the
    intercept. It is there because inhibition data is routinely
    separable -- every compound above some lipophilicity inhibits --
    and a separable logistic fit has no finite maximum, so without it
    the coefficients run off to infinity and the iteration reports a
    number that only means "it kept going". The penalty is a parameter
    and it is reported back, so a caller can see what was imposed.
    """
    n = len(X)
    if n == 0:
        raise ValueError("a fit needs data")
    if len(y) != n:
        raise ValueError("one label per compound")
    p = len(X[0]) + 1
    D = [[1.0] + [float(v) for v in row] for row in X]
    yy = [1.0 if v else 0.0 for v in y]
    b = [0.0] * p
    it = 0
    dev = 0.0
    for it in range(1, int(iters) + 1):
        eta = [_w.dot(D[i], b) for i in range(n)]
        mu = [_logistic(v) for v in eta]
        w = [max(mu[i] * (1.0 - mu[i]), 1e-10) for i in range(n)]
        z = [eta[i] + (yy[i] - mu[i]) / w[i] for i in range(n)]
        A = [[_w.csum(D[i][a] * w[i] * D[i][c] for i in range(n))
              for c in range(p)] for a in range(p)]
        for a in range(1, p):
            A[a][a] += ridge
        rhs = [_w.csum(D[i][a] * w[i] * z[i] for i in range(n))
               for a in range(p)]
        nb = _w.solve_chol(_w.chol(A), rhs)
        step = 0.0
        for a in range(p):
            d = abs(nb[a] - b[a])
            if d > step:
                step = d
        b = nb
        if step < tol:
            break
    eta = [_w.dot(D[i], b) for i in range(n)]
    mu = [_logistic(v) for v in eta]
    dev = -2.0 * _w.csum(
        (yy[i] * math.log(mu[i] if mu[i] > 1e-300 else 1e-300)
         + (1.0 - yy[i]) * math.log(1.0 - mu[i] if mu[i] < 1.0 - 1e-300
                                    else 1e-300)) for i in range(n))
    # The score, which is zero at an unpenalised optimum. Reported
    # rather than asserted, because with a ridge it is zero only up to
    # the penalty -- and a caller who sees it large knows the fit did
    # not converge whatever the iteration count says.
    score = [_w.csum(D[i][a] * (yy[i] - mu[i]) for i in range(n))
             for a in range(p)]
    return {"coefficients": b, "deviance": dev, "iterations": it,
            "score": score, "ridge": float(ridge), "n": n, "p": p}


def predict(x, coefficients):
    """The inhibition probability of one descriptor row."""
    if len(coefficients) != len(x) + 1:
        raise ValueError("the model must have one coefficient per "
                         "descriptor plus an intercept")
    z = coefficients[0] + _w.dot(list(coefficients[1:]),
                                 [float(v) for v in x])
    return _logistic(z)


def cyp450_inhibition(smiles, isozyme, model=None):
    """Descriptors for a compound against one P450 isozyme.

    Parameters
    ----------
    smiles : str
        The compound.
    isozyme : str
        One of 1A2, 2C9, 2C19, 2D6, 3A4.
    model : sequence, mapping or None
        Coefficients from ``fit``, or the dictionary ``fit`` returns.
        None means no prediction is made and the reason says so.

    Returns
    -------
    RichResult
        The descriptors, named; the probability if a model was given;
        and otherwise the reason there is none.

    References
    ----------
    Veith et al. (2009) Nature Biotechnology 27(11), 1050-1055.
    """
    if isozyme not in ISOZYMES:
        raise ValueError("the isozyme is one of " + ", ".join(ISOZYMES))
    x = descriptors(smiles)
    named = {}
    for k in range(len(NAMES)):
        named[NAMES[k]] = x[k]
    coef = None
    if model is not None:
        coef = model["coefficients"] if hasattr(model, "keys") else model
    if coef is None:
        pred = None
        reason = ("no coefficients were supplied, and none are shipped: "
                  "the screen of Veith et al. is published as a dataset "
                  "and its fitted model is not published as a table, so "
                  "any default here would be invented. Fit one on "
                  "inhibition data with this module's fit function and "
                  "pass it as model.")
    else:
        pred = predict(x, coef)
        reason = ""
    return RichResult(payload={
        "descriptors": x,
        "names": list(NAMES),
        "named": named,
        "predicted": pred,
        "inhibits": None if pred is None else bool(pred >= 0.5),
        "reason": reason,
        "isozyme": isozyme,
        "n_descriptors": len(x),
        "has_model": coef is not None,
        "method": "P450 inhibition descriptors with a caller-fitted "
                  "logistic model",
    })


def cheatsheet():
    return ("cypin: P450 inhibition for 1A2/2C9/2C19/2D6/3A4. Exact "
            "graph descriptors plus a logistic model the caller fits; "
            "no coefficients are shipped because none are published")
