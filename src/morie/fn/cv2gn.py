# morie.fn -- function file (rootcoder007/morie)
"""CV2 genomic cross-validation: both train and test lines evaluated in at least one environment."""

import math

from ._qpcore import inverse, ssum
from ._richresult import RichResult
from ._rng import random_uniform


def _pearson(a, b):
    ma, mb = ssum(a) / len(a), ssum(b) / len(b)
    sab = ssum((u - ma) * (v - mb) for u, v in zip(a, b))
    return sab / math.sqrt(ssum((u - ma) ** 2 for u in a) * ssum((v - mb) ** 2 for v in b))


def cv2_genomic(y, markers, env, n_folds=5, lam=1.0, seed=0):
    r"""CV2 cross-validation of a ridge-regression BLUP genomic predictor across environments.

    Each record is a line tested in an environment, with the line's marker
    row in ``markers``. In the CV2 scheme (Burgueno et al. 2012;
    Montesinos-Lopez et al. 2022, ch. 4) RECORDS, not lines, are assigned to
    folds, so a line whose record is held out usually has records in other
    environments in the training set -- the incomplete field trial case.
    Folds come from a Philox-driven random permutation of the records
    (stream 0 of ``seed``), ``n_folds`` of near-equal size. The predictor
    is ``y = mu + E_env + x' b`` with the environment effects unpenalised and
    ridge penalty ``lam`` on the marker effects (RR-BLUP; Meuwissen, Hayes
    and Goddard 2001). The predictive ability of fold ``f`` is the Pearson
    correlation of observed and predicted held-out records; ``pa`` is the
    mean over folds.

    References
    ----------
    Burgueno, J., de los Campos, G., Weigel, K. and Crossa, J. (2012).
    Genomic prediction of breeding values when modeling genotype x
    environment interaction using pedigree and dense molecular markers.
    *Crop Science* 52, 707-719.
    Montesinos-Lopez, O. A., Montesinos-Lopez, A. and Crossa, J. (2022).
    *Multivariate Statistical Machine Learning Methods for Genomic
    Prediction*, ch. 4. Springer.

    Examples
    --------
    >>> import math
    >>> M = [[math.sin(1.1 * i + j) for j in range(4)] for i in range(8)]
    >>> y = [sum(M[i % 8][j] * (j + 1) for j in range(4)) + (0.5 if i >= 8 else 0.0) + 0.1 * math.cos(3 * i)
    ...      for i in range(16)]
    >>> r = cv2_genomic(y, [M[i % 8] for i in range(16)], [i // 8 for i in range(16)], n_folds=4, lam=0.1)
    >>> round(r["pa"], 8)
    0.99960265
    """
    yv = [float(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    Mk = [[float(v) for v in r] for r in (markers.tolist() if hasattr(markers, "tolist") else markers)]
    ev = list(env.tolist() if hasattr(env, "tolist") else env)
    n = len(yv)
    envs = sorted(set(ev), key=str)
    D = [[1.0] + [1.0 if ev[i] == e else 0.0 for e in envs[1:]] + Mk[i] for i in range(n)]
    p = len(D[0])
    pen = [0.0] * len(envs) + [float(lam)] * len(Mk[0])
    u = [float(v) for v in random_uniform(n, seed=seed, stream=0)]
    perm = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(math.floor(u[i] * (i + 1)))
        perm[i], perm[j] = perm[j], perm[i]
    F = int(n_folds)
    fold = [0] * n
    for r, idx in enumerate(perm):
        fold[idx] = r % F
    pas = []
    for f in range(F):
        tr = [i for i in range(n) if fold[i] != f]
        te = [i for i in range(n) if fold[i] == f]
        A = inverse(
            [[ssum(D[i][a] * D[i][b] for i in tr) + (pen[a] if a == b else 0.0) for b in range(p)] for a in range(p)]
        )
        g = [ssum(D[i][a] * yv[i] for i in tr) for a in range(p)]
        beta = [ssum(A[a][b] * g[b] for b in range(p)) for a in range(p)]
        pred = [ssum(D[i][a] * beta[a] for a in range(p)) for i in te]
        pas.append(_pearson([yv[i] for i in te], pred))
    return RichResult(
        payload={
            "pa": ssum(pas) / F,
            "pa_folds": pas,
            "folds": fold,
            "n": n,
            "method": "CV2 (records held out), RR-BLUP with environment effects",
        }
    )


def cheatsheet():
    return "cv2gn: CV2 genomic cross-validation (records to folds), RR-BLUP predictive ability"


# compact alias per ledger/NAMING.md
cv2genomic = cv2_genomic
