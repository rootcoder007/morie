# morie.fn -- function file (rootcoder007/morie)
r"""Sobol global sensitivity indices, on a real Sobol design.

The variance decomposition splits :math:`\mathrm{Var}(f)` into
contributions from each input and each interaction. Two summaries
follow:

.. math:: S_i = \frac{V_i}{V}
          = \frac{\mathrm{Var}_{x_i}(E_{x_{\sim i}}[f\mid x_i])}{V},
          \qquad
          S_{T_i} = \frac{E_{x_{\sim i}}[\mathrm{Var}_{x_i}(f\mid
          x_{\sim i})]}{V},

the first-order index -- what fixing :math:`x_i` alone would remove --
and the total index, which includes every interaction :math:`x_i`
takes part in. :math:`S_i = S_{T_i}` exactly when :math:`x_i` does not
interact, and :math:`\sum_i S_i = 1` exactly when nothing interacts,
so the gap between them is the interaction structure and is reported
as such.

**The estimators** (both on :math:`A`, :math:`B` and the hybrid
matrices :math:`A_B^{(i)}`, which is :math:`A` with column :math:`i`
taken from :math:`B`):

.. math:: \hat V_i = \frac1N\sum_j f(B)_j\big(f(A_B^{(i)})_j
          - f(A)_j\big), \qquad
          \hat V_{T_i} = \frac1{2N}\sum_j\big(f(A)_j
          - f(A_B^{(i)})_j\big)^2.

The total-index estimator is Jansen's; both are unbiased and cost
:math:`N(d+2)` model runs.

**Correction to an earlier version of this module.** It generated
:math:`A` and :math:`B` as van der Corput sequences **in distinct
prime bases** while calling them "the Sobol sequence's own
low-discrepancy points". Van der Corput in distinct primes is the
**Halton** sequence, not Sobol. The design now defaults to a genuine
Sobol sequence -- Sobol (1967) with the Bratley-Fox direction numbers,
already implemented in :mod:`abcgp` -- generated in :math:`2d`
dimensions and **split by column**, the standard construction, which
gives independent :math:`A` and :math:`B` without the correlation that
continuing one sequence in the same dimensions would introduce.
``design="halton"`` keeps the old points available, and
``design="random"`` gives a pseudo-random design, so the three can be
compared rather than argued about.

References
----------
Sobol', I. M. (2001) "Global sensitivity indices for nonlinear
mathematical models and their Monte Carlo estimates", *Mathematics
and Computers in Simulation* 55(1-3), 271-280,
doi:10.1016/S0378-4754(00)00270-6. [PDF supplied by Vee.] The
ANOVA-style decomposition of the model output into terms of
increasing order, the indices S_i = V_i / V, and the total indices.

Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M. &
Tarantola, S. (2010) "Variance based sensitivity analysis of model
output. Design and estimator for the total sensitivity index",
*Computer Physics Communications* 181(2), 259-270,
doi:10.1016/j.cpc.2009.09.018. [PDF supplied by Vee.] Sec. 3-4 and
Table 2: the A / B / A_B^(i) design, with the note that the choice is
driven by the use of quasi-random sequences and that the points of A,
and hence of A_B^(i), are better distributed than those of B and B_A
when quasi-random points are used; the total-index estimator
(1/2N) sum_j (f(A)_j - f(A_B^(i))_j)^2, labelled "Jansen 1999" in
Table 2, proved more efficient than the alternatives (Theorem 4 of
ref. [38] there) and stated as "the best practice so far for S_Ti",
computed with quasi-random numbers in the A, A_B setting; the
first-order estimator (1/N) sum_j f(B)_j (f(A_B^(i))_j - f(A)_j); and
Sec. 5.1, "Using Sobol' quasi-random sequences", which lists Faure,
Niederreiter, HALTON, Hammersley and Sobol' as DISTINCT quasi-random
families -- the distinction this module previously collapsed.

Ishigami, T. & Homma, T. (1990) "An importance quantification
technique in uncertainty analysis for computer models", *Proceedings
of the First International Symposium on Uncertainty Modeling and
Analysis (ISUMA '90)*, 398-403, doi:10.1109/ISUMA.1990.151285. The
test function whose exact indices anchor this module.

Jansen, M. J. W. (1999) "Analysis of variance designs for model
output", *Computer Physics Communications* 117(1-2), 35-43,
doi:10.1016/S0010-4655(98)00154-4. The total-index estimator.

Sobol', I. M. (1967) "On the distribution of points in a cube and the
approximate evaluation of integrals", *USSR Computational Mathematics
and Mathematical Physics* 7(4), 86-112,
doi:10.1016/0041-5553(67)90144-9; Bratley, P. & Fox, B. L. (1988)
"Algorithm 659: Implementing Sobol's quasirandom sequence generator",
*ACM Transactions on Mathematical Software* 14(1), 88-100,
doi:10.1145/42288.214372. The sequence itself; implemented in
:mod:`abcgp`.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["sample_matrices", "sobol_indices", "ishigami",
           "ishigami_exact"]

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
_DESIGNS = ("sobol", "halton", "random")


def sample_matrices(N, d, design="sobol", skip=1, seed=0):
    r"""The two independent sample matrices :math:`A` and :math:`B`.

    ``sobol`` draws a :math:`2d`-dimensional Sobol sequence and splits
    it by COLUMN -- the standard construction. Continuing one
    :math:`d`-dimensional sequence to get :math:`B` would not give
    independent points: in base 2 with :math:`N` a power of two,
    points :math:`j` and :math:`j+N` share their leading bits, the
    estimator's cross terms stop cancelling, and :math:`S_i` comes out
    wrong.
    """
    n, dd = int(N), int(d)
    if n < 2 or dd < 1:
        raise ValueError("sobolI: need N >= 2 and d >= 1")
    if design not in _DESIGNS:
        raise ValueError("sobolI: design must be one of %s, got %r"
                         % (", ".join(_DESIGNS), design))
    if design == "sobol":
        from .abcgp import sobol_sequence
        pts = sobol_sequence(n, 2 * dd, skip=int(skip))
        A = [[float(pts[j][a]) for a in range(dd)] for j in range(n)]
        B = [[float(pts[j][dd + a]) for a in range(dd)]
             for j in range(n)]
    elif design == "halton":
        if 2 * dd > len(_PRIMES):
            raise ValueError("sobolI: the Halton design here has "
                             "only %d bases, so d <= %d"
                             % (len(_PRIMES), len(_PRIMES) // 2))
        A = [[k.vdc(j + int(skip), _PRIMES[a]) for a in range(dd)]
             for j in range(n)]
        B = [[k.vdc(j + int(skip), _PRIMES[dd + a])
              for a in range(dd)] for j in range(n)]
    else:
        rng = np.random.default_rng(seed)
        A = [[float(rng.uniform()) for _ in range(dd)]
             for _ in range(n)]
        B = [[float(rng.uniform()) for _ in range(dd)]
             for _ in range(n)]
    return {"A": A, "B": B, "design": design, "N": n, "d": dd,
            "model_runs": n * (dd + 2),
            "note": "a 2d-dimensional low-discrepancy sequence split "
                    "by COLUMN; the halton design is what this module "
                    "used to call 'Sobol'"}


def sobol_indices(model, input_dist=None, N=64, d=None,
                  design="sobol", skip=1, seed=0):
    r"""First-order and total indices by the Saltelli/Jansen
    estimators.

    ``model`` maps a length-:math:`d` vector on the unit cube to a
    scalar; ``input_dist`` is an optional list of per-dimension
    inverse CDFs applied first.
    """
    dd = int(d) if d is not None else (len(input_dist)
                                       if input_dist else 2)
    n = int(N)
    S_ = sample_matrices(n, dd, design=design, skip=skip, seed=seed)
    A, B = S_["A"], S_["B"]

    def tf(row):
        if input_dist is None:
            return list(row)
        return [input_dist[a](row[a]) for a in range(dd)]

    fA = [float(model(tf(A[j]))) for j in range(n)]
    fB = [float(model(tf(B[j]))) for j in range(n)]
    V = k.variance(fA + fB, 1)
    if V <= 0.0:
        raise ValueError("sobolI: the model output has zero variance, "
                         "so no index is defined")
    S, ST = [], []
    for i in range(dd):
        AB = [[B[j][a] if a == i else A[j][a] for a in range(dd)]
              for j in range(n)]
        fAB = [float(model(tf(AB[j]))) for j in range(n)]
        vi = sum(fB[j] * (fAB[j] - fA[j]) for j in range(n)) / n
        vti = sum((fA[j] - fAB[j]) ** 2 for j in range(n)) / (2.0 * n)
        S.append(vi / V)
        ST.append(vti / V)
    return RichResult(payload={
        "estimate": S[0], "S": S, "ST": ST, "V": V, "n": n, "d": dd,
        "design": S_["design"], "model_runs": S_["model_runs"],
        "sum_S": sum(S),
        "interaction": [ST[i] - S[i] for i in range(dd)],
        "additive": abs(sum(S) - 1.0) < 0.05,
        "method": "Saltelli et al. (2010) Table 2 design with the "
                  "Jansen (1999) total-index estimator, on a Sobol "
                  "(1967) sequence",
        "note": "ST - S is the interaction share; sum(S) = 1 only "
                "when the model is additive",
    })


def ishigami(x, a=7.0, b=0.1):
    r""":math:`\sin x_1 + a\sin^2 x_2 + b x_3^4\sin x_1`.

    The standard test case: :math:`x_3` has NO first-order effect at
    all yet a large total effect, so a method that confuses the two is
    caught immediately.
    """
    v = [float(q) for q in k.vec(x)]
    if len(v) != 3:
        raise ValueError("sobolI: the Ishigami function takes 3 "
                         "inputs, got %d" % len(v))
    return (math.sin(v[0]) + float(a) * math.sin(v[1]) ** 2
            + float(b) * v[2] ** 4 * math.sin(v[0]))


def ishigami_exact(a=7.0, b=0.1):
    r"""The closed-form indices, with :math:`x\sim U(-\pi,\pi)^3`.

    :math:`V = a^2/8 + b\pi^4/5 + b^2\pi^8/18 + 1/2`,
    :math:`V_1 = \tfrac12(1 + b\pi^4/5)^2`, :math:`V_2 = a^2/8`,
    :math:`V_3 = 0`, and
    :math:`V_{T_3} = 8b^2\pi^8/225`.
    """
    A, B = float(a), float(b)
    pi = math.pi
    V = (A * A / 8.0 + B * pi ** 4 / 5.0
         + B * B * pi ** 8 / 18.0 + 0.5)
    V1 = 0.5 * (1.0 + B * pi ** 4 / 5.0) ** 2
    V2 = A * A / 8.0
    V3 = 0.0
    VT3 = 8.0 * B * B * pi ** 8 / 225.0
    return {"V": V, "S": [V1 / V, V2 / V, V3 / V],
            "ST": [(V1 + VT3) / V, V2 / V, VT3 / V],
            "note": "x3 has first-order index EXACTLY zero and a "
                    "large total index -- the case that separates the "
                    "two"}


def cheatsheet():
    return ("sobolI: S_i is what fixing x_i alone would remove; S_Ti "
            "includes every interaction x_i is in. S_i = S_Ti iff x_i "
            "does not interact, and sum(S_i) = 1 iff nothing does -- so "
            "the GAP is the interaction structure. Estimators on A, B "
            "and A_B^(i) (A with column i from B): V_i = mean f(B)"
            "(f(A_B)-f(A)), V_Ti = mean (f(A)-f(A_B))^2 / 2, costing "
            "N(d+2) runs. The design must give INDEPENDENT A and B: "
            "take a 2d-dimensional Sobol sequence and split by COLUMN. "
            "Van der Corput in distinct primes is HALTON, not Sobol -- "
            "this module used to confuse the two. Anchor on Ishigami, "
            "where x3 has S3 = 0 exactly but a large total index.")


# compact alias per ledger/NAMING.md
sobolindices = sobol_indices
