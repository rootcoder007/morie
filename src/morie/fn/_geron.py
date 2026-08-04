# morie.fn -- shelf core (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Applied machine-learning shelf core.

Spec: Geron, A., *Hands-On Machine Learning with Scikit-Learn and
PyTorch*, O'Reilly (2026).  Locators are the printed page numbers of
that edition, taken from the running heads of the extracted text.

Determinism note.  The book's splitters take ``random_state=42`` and
its MC-dropout listing takes ``torch.manual_seed(42)``.  Neither
generator can be reproduced identically in R, so every routine here
uses the book's OWN deterministic alternative where it has one (the
CRC-32 identifier hash of p. 58) and otherwise takes the stochastic
part as caller-supplied input (a matrix of forward passes, a matrix of
bootstrap predictions).  Nothing here draws a random number.
"""

from __future__ import annotations

import math
import zlib

_MASK32 = 0xFFFFFFFF


def _crc32_int64(value):
    """CRC-32 of an identifier, hashed the way the book hashes it.

    p. 58 writes ``crc32(np.int64(identifier))``: the eight
    little-endian bytes of a signed 64-bit integer.
    """
    v = int(value) & 0xFFFFFFFFFFFFFFFF
    return zlib.crc32(v.to_bytes(8, "little")) & _MASK32


def _softmax(row):
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return [v / s for v in ex]


# --- ch. 4, p. 158: the bias/variance trade-off ------------------------

def bvdecomp(preds, truth, noisevar=0.0):
    """Squared-error decomposition into bias, variance and noise.

    p. 158 states IN WORDS that the generalization error is the sum of
    bias, variance and irreducible error; NO formula is printed there.
    The decomposition computed here is therefore stated explicitly
    rather than attributed to the book:

        MSE = mean_i (mean_b f_b(x_i) - y_i)^2      (bias^2)
            + mean_i var_b(f_b(x_i))               (variance)
            + noisevar                             (irreducible)

    ``preds`` is a B-by-n matrix: one row per model trained on a
    different training set, one column per test point.  Supplying the
    predictions rather than fitting anything keeps the routine free of
    the resampling randomness the book's discussion assumes.
    """
    rows = [[float(v) for v in r] for r in preds]
    b = len(rows)
    if b < 2:
        raise ValueError("need at least 2 predictor rows, got %d" % b)
    n = len(rows[0])
    if n < 1 or any(len(r) != n for r in rows):
        raise ValueError("all predictor rows must be the same non-zero length")
    y = [float(v) for v in truth]
    if len(y) != n:
        raise ValueError("truth must have one entry per test point")
    means = [sum(rows[j][i] for j in range(b)) / b for i in range(n)]
    bias2 = sum((means[i] - y[i]) ** 2 for i in range(n)) / n
    var = sum(
        sum((rows[j][i] - means[i]) ** 2 for j in range(b)) / b for i in range(n)
    ) / n
    mse = sum(
        sum((rows[j][i] - y[i]) ** 2 for j in range(b)) / b for i in range(n)
    ) / n
    noisevar = float(noisevar)
    return {
        "bias2": bias2,
        "variance": var,
        "noise": noisevar,
        "total": bias2 + var + noisevar,
        "mse": mse,
        "residual": mse - bias2 - var,
        "b": b,
        "n": n,
    }


# --- ch. 11, p. 410: Monte Carlo dropout -------------------------------

def mcdrop(logits):
    """Average the softmax over repeated stochastic forward passes.

    The listing on p. 410 is ``softmax(logits)`` over a batch repeated
    T times, then ``.mean(dim=1)`` across the T passes.  ``logits`` here
    is that same three-level structure -- n instances by T passes by k
    classes -- supplied by the caller, so the dropout masks (which the
    book seeds with ``torch.manual_seed(42)``) live outside this
    function and both language arms see identical numbers.
    """
    n = len(logits)
    if n < 1:
        raise ValueError("need at least one instance")
    t = len(logits[0])
    k = len(logits[0][0])
    probs = []
    stds = []
    ents = []
    for i in range(n):
        if len(logits[i]) != t:
            raise ValueError("every instance needs the same number of passes")
        passes = [_softmax([float(v) for v in logits[i][j]]) for j in range(t)]
        if any(len(p) != k for p in passes):
            raise ValueError("every pass needs the same number of classes")
        mean = [sum(p[c] for p in passes) / t for c in range(k)]
        sd = [
            math.sqrt(sum((p[c] - mean[c]) ** 2 for p in passes) / t)
            for c in range(k)
        ]
        probs.append(mean)
        stds.append(sd)
        ents.append(-sum(v * math.log(v) for v in mean if v > 0.0))
    top = [max(range(k), key=lambda c: probs[i][c]) for i in range(n)]
    return {
        "probs": probs,
        "sds": stds,
        "pred": top,
        "meanmaxprob": sum(probs[i][top[i]] for i in range(n)) / n,
        "meanmaxsd": sum(stds[i][top[i]] for i in range(n)) / n,
        "meanentropy": sum(ents) / n,
        "n": n,
        "t": t,
        "k": k,
    }


# --- ch. 2, pp. 58-61: splitting off a test set ------------------------

def ttsplit(ids, testratio=0.2):
    """Stable train/test split by identifier hash, p. 58.

    The book's own stable alternative to a seeded shuffle: an instance
    goes to the test set when ``crc32(int64(id)) < testratio * 2**32``.
    It is a pure function of the identifiers, which is exactly why the
    book prefers it -- and why it survives translation to R unchanged.
    """
    ids = [int(v) for v in ids]
    if not ids:
        raise ValueError("ids must be non-empty")
    if any(v < 0 or v >= 2 ** 53 for v in ids):
        raise ValueError("ids must be non-negative and below 2**53")
    testratio = float(testratio)
    if not 0.0 < testratio < 1.0:
        raise ValueError("testratio must lie strictly in (0, 1)")
    cut = testratio * 2.0 ** 32
    test = [i for i, v in enumerate(ids) if _crc32_int64(v) < cut]
    n = len(ids)
    return {
        "test": test,
        "train": [i for i in range(n) if i not in set(test)],
        "ntest": len(test),
        "ntrain": n - len(test),
        "ratio": len(test) / float(n),
        "n": n,
    }


def tvtsplit(ids, valratio=0.2, testratio=0.2):
    """Three-way train/validation/test split by identifier hash.

    p. 61 obtains a three-way split by calling the splitter twice; the
    hash generalization used here -- one CRC-32 per identifier, mapped
    to [0, 1) and cut at ``testratio`` and ``testratio + valratio`` --
    is OURS, not the book's, and is used because it stays deterministic
    across both language arms.  The p. 58 hash rule itself is the
    book's.
    """
    ids = [int(v) for v in ids]
    if not ids:
        raise ValueError("ids must be non-empty")
    if any(v < 0 or v >= 2 ** 53 for v in ids):
        raise ValueError("ids must be non-negative and below 2**53")
    valratio = float(valratio)
    testratio = float(testratio)
    if valratio <= 0.0 or testratio <= 0.0 or valratio + testratio >= 1.0:
        raise ValueError("valratio and testratio must be positive and sum below 1")
    test, val, train = [], [], []
    for i, v in enumerate(ids):
        h = _crc32_int64(v) / 2.0 ** 32
        if h < testratio:
            test.append(i)
        elif h < testratio + valratio:
            val.append(i)
        else:
            train.append(i)
    n = len(ids)
    return {
        "train": train,
        "val": val,
        "test": test,
        "ntrain": len(train),
        "nval": len(val),
        "ntest": len(test),
        "n": n,
    }


def stratsplt(strata, testratio=0.2):
    """Stratified test split by proportional allocation, pp. 60-61.

    The book's point is the GUARANTEE, not the shuffle: "the right
    number of instances are sampled from each stratum to guarantee that
    the test set is representative".  Allocation here is exactly that
    -- ``round(n_s * testratio)`` from each stratum -- with the members
    taken in their original order so the result is reproducible without
    a seed.  ``maxdev`` reports the largest gap between a stratum's
    share of the test set and its share of the population, which is the
    quantity the book checks on p. 61.
    """
    strata = [str(s) for s in strata]
    n = len(strata)
    if n < 1:
        raise ValueError("strata must be non-empty")
    testratio = float(testratio)
    if not 0.0 < testratio < 1.0:
        raise ValueError("testratio must lie strictly in (0, 1)")
    levels = sorted(set(strata))
    test = []
    for lv in levels:
        idx = [i for i in range(n) if strata[i] == lv]
        take = int(math.floor(len(idx) * testratio + 0.5))
        test.extend(idx[:take])
    test.sort()
    ntest = len(test)
    devs = []
    for lv in levels:
        pop = sum(1 for s in strata if s == lv) / float(n)
        got = (sum(1 for i in test if strata[i] == lv) / float(ntest)) if ntest else 0.0
        devs.append(abs(got - pop))
    return {
        "test": test,
        "train": [i for i in range(n) if i not in set(test)],
        "ntest": ntest,
        "ntrain": n - ntest,
        "maxdev": max(devs) if devs else 0.0,
        "nstrata": len(levels),
        "n": n,
    }


# --- ch. 12, p. 423: Equation 12-1, the convolutional layer ------------

def convlayer(x, kernel, bias=None, stride=(1, 1), padding=(0, 0)):
    """Equation 12-1, p. 423 -- output of a convolutional layer.

        z[i, j, k] = b[k] + sum_u sum_v sum_k' x[i', j', k'] w[u, v, k', k]
        with i' = i * sh + u and j' = j * sw + v

    ``x`` is height by width by in-channels, ``kernel`` is fh by fw by
    in-channels by out-channels, both as nested lists.  ``padding`` is
    the zero padding named on p. 421.  This is a cross-correlation, as
    the book's own footnote 6 on p. 419 points out.
    """
    xs = [[[float(v) for v in col] for col in row] for row in x]
    ks = [[[[float(v) for v in oc] for oc in ic] for ic in col] for col in kernel]
    h = len(xs)
    w = len(xs[0])
    cin = len(xs[0][0])
    fh = len(ks)
    fw = len(ks[0])
    if len(ks[0][0]) != cin:
        raise ValueError("kernel in-channels must match the input")
    cout = len(ks[0][0][0])
    sh, sw = int(stride[0]), int(stride[1])
    ph, pw = int(padding[0]), int(padding[1])
    if sh < 1 or sw < 1 or ph < 0 or pw < 0:
        raise ValueError("stride must be positive and padding non-negative")
    b = [0.0] * cout if bias is None else [float(v) for v in bias]
    if len(b) != cout:
        raise ValueError("bias must have one entry per output feature map")
    oh = (h + 2 * ph - fh) // sh + 1
    ow = (w + 2 * pw - fw) // sw + 1
    if oh < 1 or ow < 1:
        raise ValueError("kernel is larger than the padded input")
    out = []
    for i in range(oh):
        row = []
        for j in range(ow):
            cell = list(b)
            for u in range(fh):
                ii = i * sh + u - ph
                if not 0 <= ii < h:
                    continue
                for v in range(fw):
                    jj = j * sw + v - pw
                    if not 0 <= jj < w:
                        continue
                    for c in range(cin):
                        xv = xs[ii][jj][c]
                        if xv == 0.0:
                            continue
                        kk = ks[u][v][c]
                        for o in range(cout):
                            cell[o] += xv * kk[o]
            row.append(cell)
        out.append(row)
    flat = [v for row in out for cell in row for v in cell]
    return {
        "z": out,
        "height": oh,
        "width": ow,
        "channels": cout,
        "total": sum(flat),
        "maxz": max(flat),
        "nparams": fh * fw * cin * cout + cout,
    }


# --- ch. 12, p. 476: object tracking -----------------------------------

def trkassign(posdist, appdist=None, weight=0.5, maxn=8):
    """Minimum-cost detection-to-track assignment, p. 476.

    p. 476 describes DeepSORT in words and prints no formula, but it
    does state the objective the assignment step solves: it "finds the
    combination of mappings that minimizes the distance between the
    detections and the predicted positions of tracked objects, while
    also minimizing the appearance discrepancy".  That objective is
    what is implemented -- the combined cost
    ``(1 - weight) * position + weight * appearance`` minimized over
    all one-to-one mappings.  The Kalman prediction step and the
    appearance network are the CALLER's; this routine takes their
    output as the two cost matrices.

    ponytail: exact optimum by exhaustive search over permutations,
    capped at ``maxn`` tracks.  The Hungarian algorithm the book names
    gives the same answer in O(n^3) -- swap it in if n ever exceeds a
    handful.
    """
    pos = [[float(v) for v in r] for r in posdist]
    nt = len(pos)
    if nt < 1:
        raise ValueError("posdist must have at least one row")
    nd = len(pos[0])
    if any(len(r) != nd for r in pos):
        raise ValueError("posdist must be rectangular")
    if nt > int(maxn) or nd > int(maxn):
        raise ValueError("exhaustive assignment is capped at maxn=%d" % int(maxn))
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")
    if appdist is None:
        app = [[0.0] * nd for _ in range(nt)]
    else:
        app = [[float(v) for v in r] for r in appdist]
        if len(app) != nt or any(len(r) != nd for r in app):
            raise ValueError("appdist must match the shape of posdist")
    cost = [
        [(1.0 - weight) * pos[i][j] + weight * app[i][j] for j in range(nd)]
        for i in range(nt)
    ]
    m = min(nt, nd)
    best = None
    bestcost = float("inf")
    for rows in _combos(range(nt), m):
        for cols in _perms(range(nd), m):
            c = sum(cost[rows[q]][cols[q]] for q in range(m))
            if c < bestcost:
                bestcost = c
                best = [(rows[q], cols[q]) for q in range(m)]
    return {
        "cost": bestcost,
        "assignment": best,
        "nmatched": len(best),
        "nunmatchedtracks": nt - len(best),
        "nunmatcheddets": nd - len(best),
        "meancost": bestcost / len(best),
    }


def _combos(seq, k):
    seq = list(seq)
    if k == 0:
        yield []
        return
    for i in range(len(seq) - k + 1):
        for rest in _combos(seq[i + 1:], k - 1):
            yield [seq[i]] + rest


def _perms(seq, k):
    seq = list(seq)
    if k == 0:
        yield []
        return
    for i in range(len(seq)):
        for rest in _perms(seq[:i] + seq[i + 1:], k - 1):
            yield [seq[i]] + rest


# --- ch. 12, pp. 458-459: using a pretrained model ---------------------

def pretprep(image, size, mean, sd, logits=None, topk=1):
    """Preprocess an image for a pretrained model, pp. 458-459.

    The section's substance is the preprocessing contract, not the
    download: the image must be brought to the size the pretrained
    model expects (224 x 224 for ConvNeXt) and the pixel intensities
    standardized per colour channel "using ImageNet's means and
    standard deviations for each channel".  Those constants are NOT
    printed in the extracted text, so ``mean`` and ``sd`` are required
    arguments rather than baked-in defaults.

    ponytail: centre crop to a square, then nearest-neighbour
    subsample.  ``weights.transforms()`` uses bilinear resizing; swap
    the sampler if you need to match torchvision to the last decimal.

    Passing ``logits`` also applies the p. 459 read-out, ``argmax`` over
    the class logits, and reports the top-k classes.
    """
    img = [[[float(v) for v in px] for px in row] for row in image]
    h = len(img)
    w = len(img[0])
    c = len(img[0][0])
    size = int(size)
    if size < 1:
        raise ValueError("size must be positive")
    mean = [float(v) for v in mean]
    sd = [float(v) for v in sd]
    if len(mean) != c or len(sd) != c or any(s <= 0.0 for s in sd):
        raise ValueError("mean and sd need one positive entry per channel")
    side = min(h, w)
    r0 = (h - side) // 2
    c0 = (w - side) // 2
    out = []
    for i in range(size):
        row = []
        for j in range(size):
            si = r0 + (i * side) // size
            sj = c0 + (j * side) // size
            row.append([(img[si][sj][k] - mean[k]) / sd[k] for k in range(c)])
        out.append(row)
    flat = [v for row in out for px in row for v in px]
    res = {
        "pixels": out,
        "size": size,
        "channels": c,
        "cropside": side,
        "pixelmean": sum(flat) / len(flat),
        "pixelmax": max(flat),
    }
    if logits is not None:
        lg = [float(v) for v in logits]
        order = sorted(range(len(lg)), key=lambda i: (-lg[i], i))
        res["pred"] = order[0]
        res["topk"] = order[: int(topk)]
        res["topprob"] = _softmax(lg)[order[0]]
    return res
