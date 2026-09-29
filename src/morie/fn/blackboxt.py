# morie.fn -- function file (rootcoder007/morie)
"""Blackbox-transpose scaling: a line-for-line port of the BLACKBOXT Fortran routine of the basicspace
R package (Poole, Lewis, Rosenthal, Lo and Carroll), which places the stimuli and the respondents of a
respondent-by-stimulus rating matrix with missing entries in a common low-dimensional space."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["blackbox_transpose_fit"]

_MISS = -999.0


def _miss(v):
    return abs(v + 999.0) <= 0.001


def _jacobi(A):
    """Cyclic Jacobi eigen-decomposition of a symmetric matrix: (values descending, vector columns)."""
    n = len(A)
    a = [row[:] for row in A]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(100):
        off = 0.0
        tot = 0.0
        for i in range(n):
            tot += a[i][i] * a[i][i]
            for j in range(i + 1, n):
                off += a[i][j] * a[i][j]
        if off <= 1e-32 * tot or off == 0.0:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = a[p][q]
                if apq == 0.0:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk = a[p][k]
                    aqk = a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp = v[k][p]
                    vkq = v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: -a[i][i])
    vals = [a[i][i] for i in order]
    vecs = [[v[r][i] for i in order] for r in range(n)]
    return vals, vecs


def _pinv(a, thr):
    """Eigen pseudo-inverse sum_j z_j z_j' / w_j over |w_j| > thr (the DSYEV step of REGT/REGAT)."""
    n = len(a)
    w, z = _jacobi(a)
    return [[_dot_inv(z, w, i, kx, n, thr) for kx in range(n)] for i in range(n)]


def _dot_inv(z, w, i, kx, n, thr):
    s = 0.0
    for j in range(n):
        if abs(w[j]) > thr:
            s += z[kx][j] * (1.0 / w[j]) * z[i][j]
    return s


def _svd_v(M, ncol):
    """Right singular vectors and values of M (rows x ncol) from the Gram matrix; the largest
    entry of each vector is made positive."""
    G = [[0.0] * ncol for _ in range(ncol)]
    for a in range(ncol):
        for b in range(ncol):
            s = 0.0
            for row in M:
                s += row[a] * row[b]
            G[a][b] = s
    vals, V = _jacobi(G)
    sv = [math.sqrt(x) if x > 0 else 0.0 for x in vals]
    for j in range(ncol):
        big = 0
        for i in range(ncol):
            if abs(V[i][j]) > abs(V[big][j]):
                big = i
        if V[big][j] < 0:
            for i in range(ncol):
                V[i][j] = -V[i][j]
    return sv, V


def _corr3(x, np_, ny):
    """CORR3: pairwise-complete correlations between columns and the column sign-change vector."""
    sa = [[0.0] * ny for _ in range(ny)]
    sb = [[0.0] * ny for _ in range(ny)]
    sc = [[0.0] * ny for _ in range(ny)]
    sd = [[0.0] * ny for _ in range(ny)]
    for i in range(np_):
        xi = x[i]
        for j in range(ny):
            for jj in range(j + 1):
                # the Fortran jumps to the end of the outer loop (label 31), abandoning this j
                if _miss(xi[j]) or _miss(xi[jj]):
                    break
                sa[j][jj] += xi[j]
                if j != jj:
                    sa[jj][j] += xi[jj]
                sb[j][jj] += xi[j] * xi[j]
                if j != jj:
                    sb[jj][j] += xi[jj] * xi[jj]
                sc[j][jj] += xi[j] * xi[jj]
                sc[jj][j] = sc[j][jj]
                sd[j][jj] += 1.0
    r = [[0.0] * ny for _ in range(ny)]
    for j in range(ny):
        for jj in range(j + 1):
            aa = sd[j][jj] * sc[j][jj] - sa[j][jj] * sa[jj][j]
            bb = sd[j][jj] * sb[j][jj] - sa[j][jj] * sa[j][jj]
            cc = sd[j][jj] * sb[jj][j] - sa[jj][j] * sa[jj][j]
            r[jj][j] = 0.0 if bb * cc <= 0.0 else aa / math.sqrt(bb * cc)
            r[j][jj] = r[jj][j]
    best = -99.0
    ks = 0
    for j in range(ny):
        s = 0.0
        for jj in range(ny):
            s += abs(r[j][jj])
        if s > best:
            best = s
            ks = j
    ll = [-1 if r[ks][j] <= 0.0 else 1 for j in range(ny)]
    half = (ny - 1) // 2
    # ponytail: the Fortran always makes ny passes; a pass without a flip is a fixed point, so stop there
    for _ in range(ny):
        flipped = False
        for j in range(ny):
            kk = 0
            for jj in range(ny):
                if r[j][jj] * ll[jj] * ll[j] < 0.0:
                    kk += 1
            if kk > half:
                ll[j] = -ll[j]
                flipped = True
        if not flipped:
            break
    return ll


def _regt(np_, nf, ny, w, xs, x, psi):
    """REGT: regress each column of xs on [psi, 1] to estimate W and c; residuals go into x."""
    nf1 = nf + 1
    tsum = 0.0
    for k in range(ny):
        for j in range(nf1):
            w[k][j] = 0.0
        rows = [i for i in range(np_) if not _miss(xs[i][k]) and not _miss(psi[i][0])]
        a = [[0.0] * nf1 for _ in range(nf1)]
        for j in range(nf1):
            for jj in range(nf1):
                s = 0.0
                for i in rows:
                    s += psi[i][j] * psi[i][jj]
                a[j][jj] = s
        b = _pinv(a, 0.001)
        for i in rows:
            c = [0.0] * nf1
            for j in range(nf1):
                s = 0.0
                for jj in range(nf1):
                    s += b[j][jj] * psi[i][jj]
                c[j] = s
            for j in range(nf1):
                w[k][j] = w[k][j] + c[j] * xs[i][k]
        esum = 0.0
        for i in rows:
            s = 0.0
            for j in range(nf1):
                s += psi[i][j] * w[k][j]
            d = s - xs[i][k]
            esum += d * d
            x[i][k] = d
        w[k][nf + 1] = esum
        tsum += esum
    return tsum


def _regat(a, y, nf):
    """REGAT: [A'A]^+ A'y."""
    ns = len(y)
    b = [[0.0] * nf for _ in range(nf)]
    for j in range(nf):
        for jj in range(nf):
            s = 0.0
            for i in range(ns):
                s += a[i][j] * a[i][jj]
            b[j][jj] = s
    c = _pinv(b, 0.00001)
    bb = [[0.0] * ns for _ in range(nf)]
    for i in range(ns):
        for j in range(nf):
            s = 0.0
            for jj in range(nf):
                s += c[j][jj] * a[i][jj]
            bb[j][i] = s
    v = [0.0] * nf
    for jj in range(nf):
        s = 0.0
        for j in range(ns):
            s += bb[jj][j] * y[j]
        v[jj] = s
    return v


def _reg2t(np_, nf, ny, w, xs, x, psi, nwho):
    """REG2T: regress each row of xs minus c on W to estimate the rows of P."""
    esum = 0.0
    pxb = 0.0
    pxs = 0.0
    xns = 0.0
    for i in range(np_):
        if _miss(psi[i][0]):
            continue
        y = []
        a = []
        for j in range(ny):
            if _miss(xs[i][j]):
                continue
            y.append(xs[i][j] - w[j][nwho])
            a.append(w[j][:nf])
        v = _regat(a, y, nf)
        for k in range(ny):
            if _miss(xs[i][k]):
                continue
            s = 0.0
            for j in range(nf):
                psi[i][j] = v[j]
                s += psi[i][j] * w[k][j]
            s += w[k][nwho]
            d = s - xs[i][k]
            x[i][k] = d
            esum += d * d
        pxb += psi[i][0]
        pxs += psi[i][0] * psi[i][0]
        xns += 1.0
    pxb = pxb / xns
    pxs = pxs - xns * pxb * pxb
    return pxb, pxs, esum


def _blackbt(xb, np_, ny, nf, nfx):
    """BLACKBT: X = P W' + J c' + E by alternating least squares, then an SVD of P W'."""
    x = [row[:] for row in xb]
    xs = [row[:] for row in xb]
    xss = [row[:] for row in xb]
    ll = [0] * ny
    dc = [0.0] * ny
    ktot = 0
    svsum = 0.0
    swsum = 0.0
    for i in range(np_):
        s = 0.0
        sa = 0.0
        kk = 0
        for j in range(ny):
            if not _miss(x[i][j]):
                s += x[i][j] * x[i][j]
                sa += x[i][j]
                kk += 1
        ktot += kk
        svsum += s
        swsum += sa
        for j in range(ny):
            if not _miss(x[i][j]):
                ll[j] += 1
                dc[j] += x[i][j]
    svsum = svsum - (swsum * swsum) / ktot
    ltot = np_ * ny
    fits2 = [np_, ny, ktot, ltot - ktot, (ltot - ktot) / ltot * 100.0, svsum]
    for j in range(ny):
        dc[j] = dc[j] / ll[j]
    ll = _corr3(x, np_, ny)
    psix = [[0.0] * (nf + 1) for _ in range(np_)]
    xt = [[0.0, 0.0] for _ in range(np_)]
    w = [[0.0] * (nf + 2) for _ in range(ny)]
    for jjj in range(nf):
        xxk = 0.0
        txb = 0.0
        kkt = 0
        for i in range(np_):
            kk = 0
            s = 0.0
            for j in range(ny):
                if _miss(x[i][j]):
                    continue
                s += (x[i][j] + (-dc[j] if jjj == 0 else 0.0)) * ll[j]
                kk += 1
            xt[i][0] = _MISS
            if kk < nfx:
                continue
            wxb = s / kk
            kkt += 1
            txb += wxb
            xt[i][0] = wxb
            xxk += wxb * wxb
        txb = txb / kkt
        xxk = xxk - kkt * txb * txb
        for i in range(np_):
            if not _miss(xt[i][0]):
                xt[i][0] = xt[i][0] - txb
                xt[i][1] = 1.0
        for _ in range(4):
            _regt(np_, 1, ny, w, xs, x, xt)
            pxb, pxs, _e = _reg2t(np_, 1, ny, w, xs, x, xt, 1)
            xcor = math.sqrt(xxk / pxs)
            for i in range(np_):
                psix[i][jjj] = (xt[i][0] - pxb) * xcor
                xt[i][0] = (xt[i][0] - pxb) * xcor
                if psix[i][jjj] <= -99.0:
                    xt[i][0] = _MISS
                    psix[i][jjj] = _MISS
        last = jjj == nf - 1
        for i in range(np_):
            if last:
                psix[i][nf] = 1.0
            for j in range(ny):
                if last:
                    x[i][j] = xss[i][j]
                xs[i][j] = x[i][j]
        if not last:
            ll = _corr3(x, np_, ny)
    for _ in range(5):
        areg = _regt(np_, nf, ny, w, xs, x, psix)
        _p, _q, breg = _reg2t(np_, nf, ny, w, xs, x, psix, nf)
        for k in range(nf):
            s = 0.0
            for i in range(np_):
                s += psix[i][k]
            s = s / np_
            for i in range(np_):
                psix[i][k] = psix[i][k] - s
        if abs(areg - breg) < 0.01:
            break
    for i in range(np_):
        if _miss(psix[i][0]):
            continue
        for k in range(ny):
            s = 0.0
            for j in range(nf + 1):
                s += psix[i][j] * w[k][j]
            x[i][k] = s
    pw = [[x[i][j] - w[j][nf] for j in range(ny)] for i in range(np_)]
    # SVD of the ny x np matrix (P W')' : P = V sqrt(s), W = U sqrt(s)
    sv, V = _svd_v([[pw[i][k] for i in range(np_)] for k in range(ny)], np_)
    xdata = [[V[i][jj] * math.sqrt(sv[jj]) for jj in range(nf)] for i in range(np_)]
    wout = [[0.0] * (nf + 2) for _ in range(ny)]
    for k in range(ny):
        wout[k][0] = w[k][nf]
        for jj in range(nf):
            u = 0.0
            for i in range(np_):
                u += pw[i][k] * V[i][jj]
            u = u / sv[jj] if sv[jj] > 0 else 0.0
            wout[k][jj + 1] = u * math.sqrt(sv[jj])
        wout[k][nf + 1] = w[k][nf + 1]
    return xdata, wout, svsum, fits2


def _r2(sums):
    kjj, aa, bb, cc, dd, ee = sums
    a3 = kjj * ee - aa * bb
    b3 = kjj * cc - aa * aa
    c3 = kjj * dd - bb * bb
    return (a3 * a3) / (b3 * c3) if abs(b3 * c3) > 0.0 else 0.0


def blackbox_transpose_fit(data, missing=None, dims: int = 1) -> RichResult:
    r"""Blackbox-transpose scaling of a respondent-by-stimulus rating matrix (basicspace ``blackbox_transpose``).

    The stimuli-by-respondents matrix is decomposed as ``X = P W' + J c' + E``
    one dimension at a time by alternating least squares (starting values
    from sign-corrected row means, column signs from the pairwise-complete
    correlation matrix), refined with all dimensions jointly, and the fitted
    ``P W'`` is re-expressed by its singular value decomposition. Stimulus
    coordinates are the unit right singular vectors; respondent weights are
    ``U sqrt(s)`` with intercepts ``c``. In two dimensions the solution is
    rotated so its first axis matches the one-dimensional solution, as in
    the Fortran. Respondents with fewer than ``dims + 2`` answers are dropped
    (``None`` rows); stimuli with fewer than 8 answers are excluded from the
    starting values. Needs more scaled respondents than stimuli.

    Singular vectors are signed so that each one's largest entry is positive
    (LAPACK's signs are arbitrary), so columns can differ in sign from
    basicspace, which also rounds its coordinates to three decimals.

    Parameters
    ----------
    data : list of lists
        Respondents (rows) by stimuli (columns); ``None``/NaN or any value in
        ``missing`` is missing.
    missing : list, optional
        Missing-value codes.
    dims : int
        Number of dimensions (solutions for 1..dims are returned).

    Returns
    -------
    RichResult
        ``stimuli[d]`` rows ``[n, coord_1..coord_d, R2]``, ``individuals[d]``
        rows ``[c, w_1..w_d, R2]`` (``None`` for dropped respondents),
        ``fits`` (per dimension: SSE, SSE_explained, percent,
        cumulative_percent, R2, SE, singular), ``n_row`` (stimuli),
        ``n_col`` (scaled respondents), ``n_data``, ``n_miss``, ``ss_mean``.

    References
    ----------
    Poole, K. T. (1998). Recovering a basic space from a set of issue
    scales. *American Journal of Political Science*, 42(3), 954-993.
    Poole, K., Lewis, J., Rosenthal, H., Lo, J. and Carroll, R. (2016).
    Recovering a basic space from issue scales in R. *Journal of
    Statistical Software*, 69(7), 1-21.

    Examples
    --------
    >>> rows = [[(i * 7 + j * 3) % 10 + (j if i % 2 else -j) * 0.3 for j in range(4)] for i in range(12)]
    >>> r = blackbox_transpose_fit(rows, dims=1)
    >>> [len(r.stimuli[0]), len(r.stimuli[0][0]), r.n_col]
    [4, 3, 12]
    >>> round(sum(v[1] ** 2 for v in r.stimuli[0]), 12)
    1.0
    """
    if dims < 1:
        raise ValueError("dims must be positive")
    codes = [float(m) for m in (missing or [])]
    rows = []
    for row in data:
        out = []
        for v in row:
            bad = (
                v is None or (isinstance(v, float) and math.isnan(v)) or any(abs(float(v) - m) <= 0.001 for m in codes)
            )
            out.append(_MISS if bad else float(v))
        rows.append(out)
    n = len(rows)
    nq = len(rows[0])
    keep = [i for i in range(n) if sum(1 for v in rows[i] if not _miss(v)) >= dims + 2]
    ny = len(keep)
    if ny <= nq:
        raise ValueError("blackbox_transpose_fit needs more scaled respondents than stimuli")
    xb = [[rows[keep[j]][i] for j in range(ny)] for i in range(nq)]
    stimuli = []
    individuals = []
    rsave = []
    work5 = []
    psisave = None
    svsum = 0.0
    fits2 = None
    for kkk in range(1, dims + 1):
        xdata, w, svsum, fits2 = _blackbt(xb, nq, ny, kkk, 8)
        xt = [[0.0] * nq for _ in range(ny)]
        tot = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sume = 0.0
        for j in range(ny):
            acc = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for i in range(nq):
                s = 0.0
                for k in range(kkk):
                    s += xdata[i][k] * w[j][k + 1]
                xt[j][i] = s
                aa = s + w[j][0]
                if _miss(xb[i][j]):
                    continue
                bb = xb[i][j]
                sume += (aa - bb) * (aa - bb)
                acc[0] += 1
                acc[1] += aa
                acc[2] += bb
                acc[3] += aa * aa
                acc[4] += bb * bb
                acc[5] += aa * bb
            w[j][kkk + 1] = _r2(acc)
            for t in range(6):
                tot[t] += acc[t]
        den = tot[0] - kkk * (nq + ny) - ny
        rsave.append([sume, _r2(tot), math.sqrt(sume / den) if den > 0 and sume >= 0 else float("nan")])
        sv, V = _svd_v(xt, nq)
        work5 = sv[:]
        coords = [[V[i][jj] for jj in range(kkk)] for i in range(nq)]
        if kkk == 2:
            rot = [[0.0, 0.0], [0.0, 0.0]]
            for i in range(nq):
                for jj in range(2):
                    rot[jj][0] = rot[jj][0] + V[i][jj] * psisave[i]
            s = rot[0][0] * rot[0][0] + rot[1][0] * rot[1][0]
            for jj in range(2):
                rot[jj][0] = rot[jj][0] / math.sqrt(s)
            rot[0][1] = -rot[1][0]
            rot[1][1] = rot[0][0]
            w2 = [row[:] for row in w]
            for j in range(ny):
                for ijj in range(2):
                    s = 0.0
                    for jj in range(2):
                        s += rot[jj][ijj] * w[j][jj + 1]
                    w2[j][ijj + 1] = s
            for i in range(nq):
                for ijj in range(2):
                    s = 0.0
                    for jj in range(2):
                        s += V[i][jj] * rot[jj][ijj]
                    coords[i][ijj] = s
        stim = []
        for i in range(nq):
            acc = [0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for j in range(ny):
                s = 0.0
                for k in range(kkk):
                    s += xdata[i][k] * w[j][k + 1]
                aa = s + w[j][0]
                if _miss(xb[i][j]):
                    continue
                bb = xb[i][j]
                acc[0] += 1
                acc[1] += aa
                acc[2] += bb
                acc[3] += aa * aa
                acc[4] += bb * bb
                acc[5] += aa * bb
            stim.append([acc[0]] + coords[i] + [_r2(acc)])
        if kkk == 1:
            psisave = [V[i][0] for i in range(nq)]
        if kkk == 2:
            w = w2
        ind = [None] * n
        for ki, i in enumerate(keep):
            ind[i] = w[ki][: kkk + 2]
        stimuli.append(stim)
        individuals.append(ind)
    fits = []
    for j in range(dims):
        prev = svsum if j == 0 else rsave[j - 1][0]
        fits.append(
            {
                "SSE": rsave[j][0],
                "SSE_explained": svsum - rsave[j][0],
                "percent": (prev - rsave[j][0]) / svsum * 100.0,
                "cumulative_percent": (svsum - rsave[j][0]) / svsum * 100.0,
                "R2": rsave[j][1],
                "SE": rsave[j][2],
                "singular": work5[j],
            }
        )
    return RichResult(
        payload={
            "stimuli": stimuli,
            "individuals": individuals,
            "fits": fits,
            "n_row": fits2[0],
            "n_col": fits2[1],
            "n_data": fits2[2],
            "n_miss": fits2[3],
            "ss_mean": fits2[5],
            "dims": dims,
        }
    )


def cheatsheet() -> str:
    return "blackbox_transpose_fit(data, missing=None, dims=1) -> basicspace blackbox-transpose stimulus and respondent placements."
