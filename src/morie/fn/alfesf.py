# morie.fn -- function file (rootcoder007/morie)
r"""ESMFold confidence: decode it, fit the head, or calibrate it.

ESMFold folds from a single sequence: no MSA, no templates, just an ESM-2
language-model embedding driving a folding trunk and structure module. The
weights are 15B parameters and are not shipped here. What IS shipped is
everything the paper specifies exactly, and there are three ways in:

**Decode.** Given the head's logits -- 50 bins for LDDT, 64 for the aligned
error -- the confidence numbers are pure arithmetic. pLDDT is the
expectation of the binned distribution over bin centres. The predicted TM
score is the Zhang-Skolnick TM expectation under the error distribution,

.. math:: \mathrm{pTM} = \max_i \frac{1}{N}\sum_j \sum_b
          p^{(b)}_{ij}\,\frac{1}{1 + (d_b/d_0)^2},\qquad
          d_0 = 1.24\sqrt[3]{N-15} - 1.8,

and ipTM is the same sum restricted to pairs in different chains. No
weights are involved in any of it.

**Run a head.** Pass ``weights`` -- a matrix and bias per head -- and the
logits are computed from the trunk features first, then decoded. This is
the route when you have trained parameters, from wherever.

**Fit a head.** Pass ``features`` and observed ``lddt`` and the multinomial
logistic head is FITTED here, by full-batch gradient ascent on the
log-likelihood with an L2 penalty. Binned LDDT is a genuine multinomial
regression and there is nothing approximate about training it; the fitted
weights come back and can be fed straight into the run route.

**Calibrate.** ``temperature="fit"`` fits a single scalar dividing the
logits, the standard one-parameter recalibration, by the same gradient
ascent. A confidence head that is sharp but wrong is worse than one that is
blunt and honest, and one number fixes most of it.

The default is the decode route, because that is what the published
formulas define and it needs nothing but the model output.

References
----------
Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N. et al.
(2023) "Evolutionary-scale prediction of atomic-level protein structure
with a language model", *Science* **379**(6637), 1123-1130,
doi:10.1126/science.ade2574. ESMFold; the confidence heads follow
AlphaFold's formulation.

Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger,
O., Tunyasuvunakool, K. et al. (2021) "Highly accurate protein structure
prediction with AlphaFold", *Nature* **596**(7873), 583-589,
doi:10.1038/s41586-021-03819-3. Supplementary sections 1.9.6 (pLDDT as a
binned expectation) and 1.9.7 (pTM and the d0 above).

Zhang, Y. and Skolnick, J. (2004) "Scoring function for automated
assessment of protein structure template quality", *Proteins* **57**(4),
702-710, doi:10.1002/prot.20264. The TM score and its d0 normalisation.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017) "On calibration
of modern neural networks", *ICML* **70**, 1321-1330. Temperature
scaling, the calibration route.
"""

import math

from . import _s03core as k
from ._richresult import RichResult

__all__ = ["esmfold_confidence"]

_EPS = 1e-12
_LDDT_BINS = 50
_PAE_BINS = 64
_PAE_WIDTH = 0.5          # angstroms per bin, 0..31.5 + overflow


def _rows(x, what):
    if hasattr(x, "tolist"):
        x = x.tolist()
    out = []
    for r in x:
        if hasattr(r, "tolist"):
            r = r.tolist()
        if not isinstance(r, (list, tuple)):
            raise ValueError("%s: expected a 2-D array" % what)
        out.append([float(v) for v in r])
    if not out:
        raise ValueError("%s: empty" % what)
    return out


def _softmax_rows(M, temp=1.0):
    out = []
    for row in M:
        m = max(v / temp for v in row)
        ex = [math.exp(v / temp - m) for v in row]
        s = sum(ex)
        out.append([e / s for e in ex])
    return out


def _lddt_centres(nb):
    """Bin centres on 0..100, the AlphaFold binning."""
    return [(b + 0.5) * 100.0 / nb for b in range(nb)]


def _pae_centres(nb, width=_PAE_WIDTH):
    return [(b + 0.5) * width for b in range(nb)]


def _d0(n):
    r"""Zhang-Skolnick normalisation. Below 16 residues the cube root term
    collapses and the published clamp of 0.5 applies."""
    if n <= 15:
        return 0.5
    v = 1.24 * (n - 15.0) ** (1.0 / 3.0) - 1.8
    return v if v > 0.5 else 0.5


def _fit_multinomial(X, y, n_bins, l2=1e-3, iters=300, lr=0.5):
    """Multinomial logistic head by full-batch gradient ascent.

    Deterministic: zero init, fixed step, fixed iteration count, so the two
    language arms walk the same path. No line search, because a search that
    branches on a floating-point comparison is exactly what makes two
    implementations disagree in the last digits.
    """
    n, d = len(X), len(X[0])
    W = [[0.0] * n_bins for _ in range(d)]
    b = [0.0] * n_bins
    for _ in range(int(iters)):
        gW = [[0.0] * n_bins for _ in range(d)]
        gb = [0.0] * n_bins
        for i in range(n):
            z = [sum(X[i][a] * W[a][c] for a in range(d)) + b[c]
                 for c in range(n_bins)]
            m = max(z)
            ex = [math.exp(v - m) for v in z]
            s = sum(ex)
            p = [e / s for e in ex]
            for c in range(n_bins):
                g = (1.0 if c == y[i] else 0.0) - p[c]
                gb[c] += g
                for a in range(d):
                    gW[a][c] += g * X[i][a]
        for c in range(n_bins):
            b[c] += lr * gb[c] / n
            for a in range(d):
                W[a][c] += lr * (gW[a][c] / n - l2 * W[a][c])
    return W, b


def _fit_temperature(L, y, iters=200, lr=0.5):
    """One scalar, fitted by gradient ascent on the log-likelihood."""
    logt = 0.0
    n = len(L)
    for _ in range(int(iters)):
        g = 0.0
        t = math.exp(logt)
        for i in range(n):
            z = [v / t for v in L[i]]
            m = max(z)
            ex = [math.exp(v - m) for v in z]
            s = sum(ex)
            p = [e / s for e in ex]
            # d/dlogt of the log-likelihood
            zc = L[i][y[i]] / t
            g += -zc + sum(p[c] * L[i][c] / t for c in range(len(p)))
        logt += lr * g / n
    return math.exp(logt)


def esmfold_confidence(lddt_logits=None, pae_logits=None, features=None,
                       weights=None, lddt=None, chain_id=None,
                       temperature=1.0, l2=1e-3, iters=300, lr=0.5,
                       pae_bin_width=_PAE_WIDTH):
    r"""Decode, run, fit or calibrate the ESMFold confidence heads.

    Parameters
    ----------
    lddt_logits : array-like (n, 50), optional
        Per-residue LDDT head logits. The decode route.
    pae_logits : array-like (n*n, 64) or (n, n, 64), optional
        Aligned-error logits. Needed for pTM and ipTM.
    features : array-like (n, d), optional
        Trunk features. With ``weights`` the logits are computed from
        these; with ``lddt`` they are the design matrix for FITTING.
    weights : dict, optional
        ``{"W": (d, 50), "b": (50,)}``. Supply trained parameters here --
        nothing is bundled, and nothing is invented when they are absent.
    lddt : array-like (n,), optional
        Observed LDDT in 0..100. Providing it with ``features`` selects
        the training route and returns fitted weights.
    chain_id : array-like (n,), optional
        Chain label per residue. Required for ipTM, which is the pTM sum
        restricted to pairs in DIFFERENT chains -- with one chain it is
        not defined and is returned as None rather than as pTM.
    temperature : float or "fit"
        Divides the logits. ``"fit"`` estimates it from ``lddt``.

    Returns
    -------
    RichResult
        ``plddt`` per residue and its mean, ``ptm``, ``iptm``, ``pae``,
        the ``route`` actually taken, and any fitted ``weights`` /
        ``temperature``.
    """
    route = None
    fitted_W = fitted_b = None
    temp_used = None

    # ---- training route: features + observed lddt -> a fitted head
    if features is not None and lddt is not None:
        X = _rows(features, "alfesf features")
        obs = [float(v) for v in
               (lddt.tolist() if hasattr(lddt, "tolist") else lddt)]
        if len(obs) != len(X):
            raise ValueError("alfesf: %d feature rows but %d lddt values"
                             % (len(X), len(obs)))
        for v in obs:
            if not 0.0 <= v <= 100.0:
                raise ValueError("alfesf: lddt must be on 0..100, got %g" % v)
        y = [min(int(v / 100.0 * _LDDT_BINS), _LDDT_BINS - 1) for v in obs]
        fitted_W, fitted_b = _fit_multinomial(X, y, _LDDT_BINS,
                                              l2=l2, iters=iters, lr=lr)
        lddt_logits = [[sum(X[i][a] * fitted_W[a][c]
                            for a in range(len(X[0]))) + fitted_b[c]
                        for c in range(_LDDT_BINS)] for i in range(len(X))]
        route = "fitted a multinomial LDDT head from features and observations"

    # ---- run route: features + supplied weights -> logits
    elif features is not None and weights is not None:
        X = _rows(features, "alfesf features")
        W = _rows(weights["W"], "alfesf weights W")
        b = [float(v) for v in weights["b"]]
        if len(W) != len(X[0]):
            raise ValueError("alfesf: weights W has %d rows but the features "
                             "have %d columns" % (len(W), len(X[0])))
        if len(b) != len(W[0]):
            raise ValueError("alfesf: bias length %d does not match the %d "
                             "output bins" % (len(b), len(W[0])))
        lddt_logits = [[sum(X[i][a] * W[a][c] for a in range(len(W))) + b[c]
                        for c in range(len(b))] for i in range(len(X))]
        route = "ran a supplied LDDT head over the features"

    elif features is not None:
        raise ValueError(
            "alfesf: features were given with neither `weights` to run nor "
            "`lddt` to fit. Nothing is bundled and nothing will be invented: "
            "supply trained parameters, or observations to train on, or pass "
            "the head's logits directly.")

    if lddt_logits is None and pae_logits is None:
        raise ValueError(
            "alfesf: give lddt_logits and/or pae_logits to decode, or "
            "features with weights (to run) or with lddt (to fit).")

    if route is None:
        route = "decoded supplied logits"

    # ---- temperature
    if isinstance(temperature, str):
        if temperature != "fit":
            raise ValueError("alfesf: temperature must be a number or 'fit'")
        if lddt is None or lddt_logits is None:
            raise ValueError("alfesf: temperature='fit' needs observed lddt "
                             "and the logits to calibrate")
        obs = [float(v) for v in
               (lddt.tolist() if hasattr(lddt, "tolist") else lddt)]
        y = [min(int(v / 100.0 * _LDDT_BINS), _LDDT_BINS - 1) for v in obs]
        temp_used = _fit_temperature(lddt_logits, y)
        route += "; temperature calibrated"
    else:
        temp_used = float(temperature)
        if not temp_used > 0.0:
            raise ValueError("alfesf: temperature must be positive")

    # ---- pLDDT: the expectation of the binned distribution
    plddt = None
    if lddt_logits is not None:
        L = _rows(lddt_logits, "alfesf lddt_logits")
        nb = len(L[0])
        cen = _lddt_centres(nb)
        P = _softmax_rows(L, temp_used)
        plddt = [sum(P[i][c] * cen[c] for c in range(nb))
                 for i in range(len(P))]

    # ---- pTM / ipTM
    ptm = iptm = None
    pae = None
    if pae_logits is not None:
        raw = pae_logits.tolist() if hasattr(pae_logits, "tolist") \
            else pae_logits
        if raw and isinstance(raw[0], (list, tuple)) and raw[0] and \
                isinstance(raw[0][0], (list, tuple)):
            n = len(raw)
            flat = [raw[i][j] for i in range(n) for j in range(n)]
        else:
            flat = _rows(raw, "alfesf pae_logits")
            n = int(round(math.sqrt(len(flat))))
            if n * n != len(flat):
                raise ValueError("alfesf: %d aligned-error rows is not a "
                                 "square number of residue pairs" % len(flat))
        nb = len(flat[0])
        cen = _pae_centres(nb, pae_bin_width)
        Pp = _softmax_rows(flat, temp_used)
        pae = [[sum(Pp[i * n + j][c] * cen[c] for c in range(nb))
                for j in range(n)] for i in range(n)]
        d0 = _d0(n)
        f = [1.0 / (1.0 + (cen[c] / d0) ** 2) for c in range(nb)]
        per_i = []
        for i in range(n):
            per_i.append(sum(sum(Pp[i * n + j][c] * f[c] for c in range(nb))
                             for j in range(n)) / n)
        ptm = max(per_i)
        if chain_id is not None:
            ch = list(chain_id.tolist() if hasattr(chain_id, "tolist")
                      else chain_id)
            if len(ch) != n:
                raise ValueError("alfesf: %d chain labels for %d residues"
                                 % (len(ch), n))
            if len(set(ch)) > 1:
                inter = []
                for i in range(n):
                    js = [j for j in range(n) if ch[j] != ch[i]]
                    if not js:
                        continue
                    inter.append(sum(sum(Pp[i * n + j][c] * f[c]
                                         for c in range(nb))
                                     for j in js) / len(js))
                iptm = max(inter) if inter else None

    return RichResult(payload={
        "estimate": (sum(plddt) / len(plddt)) if plddt else ptm,
        "plddt": plddt,
        "plddt_mean": (sum(plddt) / len(plddt)) if plddt else None,
        "ptm": ptm,
        "iptm": iptm,
        "pae": pae,
        "d0": _d0(len(pae)) if pae else None,
        "temperature": temp_used,
        "weights": ({"W": fitted_W, "b": fitted_b}
                    if fitted_W is not None else None),
        "route": route,
        "n_lddt_bins": len(lddt_logits[0]) if lddt_logits is not None else None,
        "n_pae_bins": len(pae[0]) if pae else None,
        "method": ("ESMFold/AlphaFold confidence: pLDDT as the expectation "
                   "of the binned LDDT distribution, pTM as the "
                   "Zhang-Skolnick TM expectation under the aligned-error "
                   "distribution with d0 = 1.24 (N-15)^(1/3) - 1.8, ipTM "
                   "restricted to inter-chain pairs"),
        "note": ("route says which of the four paths ran. No network "
                 "weights are bundled: supply them, fit them here from "
                 "observed LDDT, or pass logits straight from a model you "
                 "ran elsewhere. iptm is None when chain_id is absent or "
                 "names a single chain -- reporting ptm in its place would "
                 "be wrong, since ipTM is by definition the inter-chain "
                 "restriction."),
    })


def cheatsheet():
    return ("alfesf: esmfold_confidence(lddt_logits, pae_logits) -> pLDDT, "
            "pTM, ipTM; or features+weights to run, features+lddt to fit "
            "(Lin et al. 2023 Science 379:1123; Jumper et al. 2021 SI 1.9.6-7)")


# compact alias per ledger/NAMING.md
esmfold_lm_only = esmfold_confidence
