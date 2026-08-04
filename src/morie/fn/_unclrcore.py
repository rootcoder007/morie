# morie.fn -- shared core for the "unclear-attribution" batch (rootcoder007/morie)
"""Computations shared by the modules resolved in the unclear-attribution batch.

Every routine here is plain Python arithmetic: no array library, no
external dependency, fixed iteration counts, no tolerance-driven early
exit.  The R mirror in ``R/unclr.R`` is a function-for-function
transcription of this file.

Sources are named per routine.  Where a routine is a textbook-standard
quantity with no single owning source it says so and names none.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------
# small linear algebra (standard; no owning source)
# --------------------------------------------------------------------


def _as_mat(M):
    return [[float(v) for v in row] for row in M]


def _as_vec(v):
    return [float(t) for t in v]


def matmul(A, B):
    """Standard matrix product."""
    n, k, m = len(A), len(B), len(B[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        oi = out[i]
        for t in range(k):
            a = Ai[t]
            if a == 0.0:
                continue
            Bt = B[t]
            for j in range(m):
                oi[j] += a * Bt[j]
    return out


def matvec(A, x):
    """Standard matrix-vector product."""
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def jacobi_eigh(M, sweeps=60):
    """Cyclic Jacobi eigendecomposition of a symmetric matrix.

    Standard method (no single owning source).  ``sweeps`` is fixed, so
    the result does not depend on a convergence tolerance.  Eigenvectors
    are sign-fixed: the entry of largest magnitude in each column is made
    positive, which removes the sign ambiguity entirely.

    Returns ``(values, vectors)`` with ``vectors[i][j]`` the i-th entry of
    the j-th eigenvector, ascending in eigenvalue.
    """
    A = _as_mat(M)
    n = len(A)
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p][q]
                if apq == 0.0:
                    continue
                theta = (A[q][q] - A[p][p]) / (2.0 * apq)
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p] = c * akp - s * akq
                    A[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k] = c * apk - s * aqk
                    A[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
    vals = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: vals[i])
    vals = [vals[i] for i in order]
    V = [[V[i][j] for j in order] for i in range(n)]
    for j in range(n):
        big = max(range(n), key=lambda i: abs(V[i][j]))
        if V[big][j] < 0:
            for i in range(n):
                V[i][j] = -V[i][j]
    return vals, V


def solve(A, b, ridge=0.0):
    """Gaussian elimination with partial pivoting; standard."""
    n = len(A)
    M = [[float(A[i][j]) + (ridge if i == j else 0.0) for j in range(n)] + [float(b[i])] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-300:
            raise ValueError("singular system; supply a ridge or a full-rank input")
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for j in range(c, n + 1):
            M[c][j] /= pv
        for r in range(n):
            if r == c:
                continue
            f = M[r][c]
            if f == 0.0:
                continue
            for j in range(c, n + 1):
                M[r][j] -= f * M[c][j]
    return [M[i][n] for i in range(n)]


def dft_amp(x):
    """Amplitude spectrum by direct DFT.

    Standard discrete Fourier transform.
    ponytail: O(n^2) transform, swap in a radix-2 FFT if inputs get long.
    """
    n = len(x)
    amps = []
    for k in range(n):
        re = im = 0.0
        for t in range(n):
            ang = -2.0 * math.pi * k * t / n
            re += x[t] * math.cos(ang)
            im += x[t] * math.sin(ang)
        amps.append(math.sqrt(re * re + im * im))
    return amps


def _mean(v):
    return sum(v) / len(v)


def _var(v, ddof=1):
    m = _mean(v)
    return sum((t - m) ** 2 for t in v) / (len(v) - ddof)


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _Phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# ====================================================================
# Lawson, A. B. (2021), Using R for Bayesian Spatial and Spatio-Temporal
# Health Modeling, CRC Press.  Equation numbers as printed.
# ====================================================================


def likprod(dens):
    """L(y|theta) = prod_i f(y_i|theta)   (Lawson eq. 3.1 p.--).

    The joint likelihood of conditionally independent observations is
    the product of the individual contributions.
    """
    d = _as_vec(dens)
    if any(t < 0 for t in d):
        raise ValueError("densities must be non-negative")
    prod = 1.0
    for t in d:
        prod *= t
    ll = sum(math.log(t) for t in d) if all(t > 0 for t in d) else float("-inf")
    return {"likelihood": prod, "loglik": ll, "n": len(d)}


def loglksum(dens):
    """l(y|theta) = sum_i log f(y_i|theta)   (Lawson eq. 3.2).

    The log of eq. (3.1); the sum form is what is actually evaluated
    because the product underflows for even moderate n.
    """
    d = _as_vec(dens)
    if any(t <= 0 for t in d):
        raise ValueError("densities must be strictly positive to take logs")
    terms = [math.log(t) for t in d]
    return {"loglik": sum(terms), "terms": terms, "n": len(d)}


def postres(y, e, theta_draws):
    """r_i = y_i - (1/G) sum_g e_i theta_i^(g)   (Lawson eq. 5.2).

    Posterior-averaged Poisson residual for tract counts with expected
    count e_i.  ``theta_draws[g][i]`` is draw g of the relative risk for
    tract i; the average is taken over the posterior sample, which is
    what distinguishes (5.2) from a plug-in residual.
    """
    yv, ev = _as_vec(y), _as_vec(e)
    T = _as_mat(theta_draws)
    G = len(T)
    if G == 0:
        raise ValueError("need at least one posterior draw")
    n = len(yv)
    if len(ev) != n or any(len(row) != n for row in T):
        raise ValueError("y, e and each posterior draw must have the same length")
    fitted = [sum(ev[i] * T[g][i] for g in range(G)) / G for i in range(n)]
    return {"residual": [yv[i] - fitted[i] for i in range(n)], "fitted": fitted, "n": n, "n_draws": G}


def intmod(lam0, lam1):
    """lambda(s|psi) = lambda_0(s|psi_0) . lambda_1(s|psi_1)   (Lawson eq. 6.3).

    Modulated point-process intensity: a population-at-risk factor times
    an excess-risk factor.  The factorisation is what lets the at-risk
    nuisance be conditioned out.
    """
    a, b = _as_vec(lam0), _as_vec(lam1)
    if len(a) != len(b):
        raise ValueError("lam0 and lam1 must have the same length")
    if any(t < 0 for t in a) or any(t < 0 for t in b):
        raise ValueError("intensities must be non-negative")
    lam = [a[i] * b[i] for i in range(len(a))]
    return {"intensity": lam, "total": sum(lam), "n": len(lam)}


def cclogl(eta, y):
    """L = prod_i {exp(eta_i)}^{y_i} / (1 + exp(eta_i))   (Lawson eq. 6.6).

    Case-control likelihood for a case event model, which reduces to a
    logistic likelihood: the at-risk population function drops out.
    Returned on the log scale, with the case probabilities.
    """
    e, yy = _as_vec(eta), _as_vec(y)
    if len(e) != len(yy):
        raise ValueError("eta and y must have the same length")
    if any(t not in (0.0, 1.0) for t in yy):
        raise ValueError("y must be 0/1 (control/case)")
    p = [1.0 / (1.0 + math.exp(-t)) for t in e]
    ll = sum(yy[i] * e[i] - math.log1p(math.exp(e[i])) for i in range(len(e)))
    return {"loglik": ll, "p": p, "n_cases": int(sum(yy)), "n": len(e)}


def mlogitlp(f, g, R):
    """logit(p_i) = f_i + g_i + R_i   (Lawson eq. 6.8).

    Contextual multilevel logit: an individual-predictor term, a
    spatial-unit covariate term, and a spatial-unit random effect.
    """
    fv, gv, rv = _as_vec(f), _as_vec(g), _as_vec(R)
    if not (len(fv) == len(gv) == len(rv)):
        raise ValueError("f, g and R must have the same length")
    eta = [fv[i] + gv[i] + rv[i] for i in range(len(fv))]
    return {"eta": eta, "p": [1.0 / (1.0 + math.exp(-t)) for t in eta], "n": len(eta)}


def lgcpint(lam0, beta, S):
    """lambda(s) = lambda_0(s) exp{beta + S(s)}   (Lawson eq. 6.18).

    First-order intensity of the Diggle et al. (1998) log-Gaussian Cox
    process: a modulating baseline, a non-zero mean level beta, and a
    zero-mean Gaussian process S(s) supplied by the caller.
    """
    a, s = _as_vec(lam0), _as_vec(S)
    if len(a) != len(s):
        raise ValueError("lam0 and S must have the same length")
    b = float(beta)
    lam = [a[i] * math.exp(b + s[i]) for i in range(len(a))]
    return {"intensity": lam, "total": sum(lam), "beta": b, "n": len(lam)}


def facrisk(alpha0, W, phi):
    """log(theta_i) = alpha_0 + sum_l w_il phi_l   (Lawson eq. 11.1).

    Poisson risk built from L unobserved components phi with area
    weights W; y_i ~ Pois(e_i theta_i).
    """
    Wm, p = _as_mat(W), _as_vec(phi)
    if any(len(row) != len(p) for row in Wm):
        raise ValueError("each row of W must have one weight per component")
    lr = [float(alpha0) + sum(row[l] * p[l] for l in range(len(p))) for row in Wm]
    return {"logrisk": lr, "risk": [math.exp(t) for t in lr], "n": len(lr), "n_components": len(p)}


def mvfacmu(e, lam, f):
    """mu_ik = e_ik rho_ik,  log(rho_ik) = lambda_k f_i   (Lawson eq. 14.1).

    Shared spatial factor across k diseases: one common factor f_i per
    area, one loading lambda_k per disease.  ``e[i][k]`` are expected
    counts; the returned matrices are row-major, area by disease.
    """
    E, lv, fv = _as_mat(e), _as_vec(lam), _as_vec(f)
    if len(E) != len(fv) or any(len(row) != len(lv) for row in E):
        raise ValueError("e must be n areas by k diseases, matching f and lam")
    rho = [[math.exp(lv[k] * fv[i]) for k in range(len(lv))] for i in range(len(fv))]
    mu = [[E[i][k] * rho[i][k] for k in range(len(lv))] for i in range(len(fv))]
    return {"rho": rho, "mu": mu, "n": len(fv), "n_disease": len(lv)}


def mlpois(beta0, beta1, age, race_effect, v, W):
    """log(lambda_i) = b0 + b1 age_i + beta(race_i) + v_i + W_i   (Lawson eq. 15.2).

    Multilevel Poisson log-rate: a fixed age slope, a categorical race
    effect already resolved to a per-observation value, an unstructured
    effect v and a spatially structured effect W.
    """
    a, r, vv, ww = _as_vec(age), _as_vec(race_effect), _as_vec(v), _as_vec(W)
    if not (len(a) == len(r) == len(vv) == len(ww)):
        raise ValueError("age, race_effect, v and W must have the same length")
    lr = [float(beta0) + float(beta1) * a[i] + r[i] + vv[i] + ww[i] for i in range(len(a))]
    return {"lograte": lr, "rate": [math.exp(t) for t in lr], "n": len(lr)}


def menorm(beta0, beta1, x_true, tau):
    """y_i ~ N(mu_i, tau^-1),  mu_i = b0 + b1 x^T_i   (Lawson eq. 16.1).

    The outcome half of the classical measurement-error model: the
    regression is on the unobserved true covariate x^T, not on the
    error-prone x.  Returns the mean and the precision-implied scale.
    """
    xt = _as_vec(x_true)
    t = float(tau)
    if t <= 0:
        raise ValueError("tau is a precision and must be positive")
    mu = [float(beta0) + float(beta1) * v for v in xt]
    return {"mu": mu, "var": 1.0 / t, "sd": 1.0 / math.sqrt(t), "n": len(mu)}


def logitre(gamma0, gamma1, d, gamma2, x, R):
    """logit(p_i) = g0 + g1 d_i + g2 x_i + R_i   (Lawson eq. 17.1).

    Binary spatial regression with an exposure d, a covariate x and a
    spatially referenced random effect R.
    """
    dv, xv, rv = _as_vec(d), _as_vec(x), _as_vec(R)
    if not (len(dv) == len(xv) == len(rv)):
        raise ValueError("d, x and R must have the same length")
    eta = [float(gamma0) + float(gamma1) * dv[i] + float(gamma2) * xv[i] + rv[i] for i in range(len(dv))]
    return {"eta": eta, "p": [1.0 / (1.0 + math.exp(-t)) for t in eta], "n": len(eta)}


def epiar(beta0, beta1, i_lag, b1):
    """log(f) = b0 + b1 log(I_{i,j-1}) + b1i   (Lawson eq. 18.3).

    Epidemic transmission term: log-linear in the previous period's own
    infective count, with a spatially referenced random effect b1i
    (ICAR prior in the book).
    """
    il, bb = _as_vec(i_lag), _as_vec(b1)
    if len(il) != len(bb):
        raise ValueError("i_lag and b1 must have the same length")
    if any(t <= 0 for t in il):
        raise ValueError("lagged infective counts must be positive to take logs")
    lf = [float(beta0) + float(beta1) * math.log(il[i]) + bb[i] for i in range(len(il))]
    return {"logf": lf, "f": [math.exp(t) for t in lf], "n": len(lf)}


def epiarnb(beta0, beta1, i_lag, nb_lag, b1):
    """log(f) = b0 + b1 log(I_{i,j-1} + sum_{l in delta_i} I_{l,j-1}) + b1i   (eq. 18.4).

    Eq. (18.3) with the neighbourhood term added: ``nb_lag[i]`` is the
    already-summed lagged infective count of the regions adjacent to i.
    """
    il, nb, bb = _as_vec(i_lag), _as_vec(nb_lag), _as_vec(b1)
    if not (len(il) == len(nb) == len(bb)):
        raise ValueError("i_lag, nb_lag and b1 must have the same length")
    tot = [il[i] + nb[i] for i in range(len(il))]
    if any(t <= 0 for t in tot):
        raise ValueError("own plus neighbour lagged counts must be positive")
    lf = [float(beta0) + float(beta1) * math.log(tot[i]) + bb[i] for i in range(len(il))]
    return {"logf": lf, "f": [math.exp(t) for t in lf], "total_lag": tot, "n": len(lf)}


# ====================================================================
# Deshmukh, S. R. & Kashikar, A. S. (2021), Probability Theory: An
# Introduction Using R, CRC Press.  Equation numbers as printed.
# ====================================================================


def cfinvpmf(t, phi_re, phi_im, x):
    """p_x = (1/2pi) int_{-pi}^{pi} e^{-itx} phi_X(t) dt   (Deshmukh eq. 4.9).

    Fourier inversion of the characteristic function of an
    integer-valued random variable.  ``t`` is the quadrature grid on
    [-pi, pi] and ``phi_re``/``phi_im`` are phi_X evaluated on it; the
    integral is taken by the trapezoid rule on the supplied grid, so the
    number of nodes -- not a tolerance -- fixes the accuracy.
    """
    tv, pr, pi_ = _as_vec(t), _as_vec(phi_re), _as_vec(phi_im)
    if not (len(tv) == len(pr) == len(pi_)):
        raise ValueError("t, phi_re and phi_im must have the same length")
    if len(tv) < 2:
        raise ValueError("need at least two quadrature nodes")
    out = []
    for xv in _as_vec(x):
        # Re[e^{-i t x} phi(t)] = cos(tx) Re phi + sin(tx) Im phi
        g = [math.cos(tv[j] * xv) * pr[j] + math.sin(tv[j] * xv) * pi_[j] for j in range(len(tv))]
        integ = sum((g[j] + g[j + 1]) * (tv[j + 1] - tv[j]) / 2.0 for j in range(len(tv) - 1))
        out.append(integ / (2.0 * math.pi))
    return {"pmf": out, "x": _as_vec(x), "n_nodes": len(tv)}


def indevk(p, joint):
    """P(A_i1 ... A_ir) = prod_l P(A_il) for every 2 <= r <= k   (Deshmukh eq. 5.1).

    Independence of k events requires every one of the
    2^k - k - 1 subset conditions to hold, not just pairwise ones.
    ``joint[m]`` is P of the intersection of the events whose bits are
    set in the mask m, for m = 0 .. 2^k - 1; masks with fewer than two
    bits set are ignored.
    """
    pv = _as_vec(p)
    k = len(pv)
    jv = _as_vec(joint)
    if len(jv) != 2**k:
        raise ValueError(f"joint must hold one probability per subset mask, i.e. {2 ** k} entries")
    n_cond = 2**k - k - 1
    worst = 0.0
    for m in range(2**k):
        bits = [i for i in range(k) if m >> i & 1]
        if len(bits) < 2:
            continue
        prod = 1.0
        for i in bits:
            prod *= pv[i]
        worst = max(worst, abs(jv[m] - prod))
    return {"n_conditions": n_cond, "max_deviation": worst, "independent": worst <= 1e-12, "k": k}


def indrv2(joint):
    """P[X1 in S1, X2 in S2] = P[X1 in S1] P[X2 in S2]   (Deshmukh eq. 5.3).

    Independence of two random variables as factorisation of the joint
    distribution into its marginals.  ``joint`` is a probability table
    over a finite partition of the two ranges; the marginals are its row
    and column sums, and the deviation reported is the largest absolute
    departure from the product.
    """
    J = _as_mat(joint)
    tot = sum(sum(r) for r in J)
    if abs(tot - 1.0) > 1e-9:
        raise ValueError(f"joint probabilities must sum to 1, got {tot}")
    rows = [sum(r) for r in J]
    cols = [sum(J[i][j] for i in range(len(J))) for j in range(len(J[0]))]
    worst = 0.0
    for i in range(len(J)):
        for j in range(len(J[0])):
            worst = max(worst, abs(J[i][j] - rows[i] * cols[j]))
    return {
        "max_deviation": worst,
        "independent": worst <= 1e-12,
        "margin_row": rows,
        "margin_col": cols,
    }


def limsupio(dev, k):
    """D_k = int_{n>=1} un_{m>=n} {|X_m - X| >= 1/k}   (Deshmukh eq. 6.1).

    The limit-superior event: |X_n - X| >= 1/k infinitely often.  On a
    realised path of finite length the event is decided by the tail, so
    a path is in D_k exactly when its last observation still exceeds the
    threshold.  ``dev[r][m]`` is |X_m - X| on path r.
    """
    D = _as_mat(dev)
    kk = int(k)
    if kk < 1:
        raise ValueError("k must be a positive integer")
    thr = 1.0 / kk
    flags = [1 if row[-1] >= thr else 0 for row in D]
    return {
        "threshold": thr,
        "in_event": flags,
        "prob": sum(flags) / len(flags),
        "n_paths": len(D),
    }


def degencdf(x, mu):
    """P[Xbar_n <= x] -> 0 if x < mu, 1 if x > mu   (Deshmukh eq. 10.3).

    Limiting distribution function of the sample mean of iid variables
    with finite mean mu: by Khintchine's WLLN the sample mean converges
    in probability, hence in law, to the degenerate law at mu.  The book
    is explicit that the limit at x = mu is not determined by the given
    information, so it is returned as NaN rather than guessed at 1/2.
    """
    m = float(mu)
    out = []
    for v in _as_vec(x):
        out.append(0.0 if v < m else (1.0 if v > m else float("nan")))
    return {"cdf": out, "mu": m, "x": _as_vec(x)}


# ====================================================================
# Klein, D. J. & Randic, M. (1993), Resistance distance, Journal of
# Mathematical Chemistry 12, 81-95.
# ====================================================================


def lappinv(A, tol=1e-9):
    """Moore-Penrose pseudoinverse of the graph Laplacian.

    L = D - A; L is singular because the all-ones vector spans its
    kernel on a connected graph, so L^+ is formed from the spectral
    decomposition with the zero modes dropped.  L^+ is basis
    independent, so it does not depend on the eigenvector convention.
    """
    Am = _as_mat(A)
    n = len(Am)
    if any(len(r) != n for r in Am):
        raise ValueError("A must be square")
    for i in range(n):
        for j in range(i + 1, n):
            if abs(Am[i][j] - Am[j][i]) > 1e-12:
                raise ValueError("A must be symmetric")
    L = [[(sum(Am[i]) if i == j else 0.0) - Am[i][j] for j in range(n)] for i in range(n)]
    vals, V = jacobi_eigh(L)
    Lp = [[0.0] * n for _ in range(n)]
    kept = 0
    for m in range(n):
        if abs(vals[m]) <= tol:
            continue
        kept += 1
        inv = 1.0 / vals[m]
        for i in range(n):
            vi = V[i][m] * inv
            for j in range(n):
                Lp[i][j] += vi * V[j][m]
    return {"Lplus": Lp, "eigenvalues": vals, "rank": kept, "n": n}


def resdist(A, tol=1e-9):
    """R_ij = L^+_ii + L^+_jj - 2 L^+_ij   (Klein & Randic 1993).

    Effective resistance between every pair of nodes when each edge is a
    unit conductance.  Unlike the shortest-path distance it falls when a
    parallel route is added, which is the property the paper is about.
    """
    lp = lappinv(A, tol)
    Lp, n = lp["Lplus"], lp["n"]
    R = [[Lp[i][i] + Lp[j][j] - 2.0 * Lp[i][j] for j in range(n)] for i in range(n)]
    return {"R": R, "Lplus": Lp, "n": n, "rank": lp["rank"]}


def commdist(A, tol=1e-9):
    """C_ij = 2m (L^+_ii + L^+_jj - 2 L^+_ij) = 2m R_ij.

    Expected commute time of the simple random walk, which is the
    resistance distance scaled by twice the total edge weight (Chandra
    et al. 1989; the resistance identity is Klein & Randic 1993).
    """
    rd = resdist(A, tol)
    m2 = sum(sum(r) for r in _as_mat(A))  # 2m for a symmetric weight matrix
    C = [[m2 * v for v in row] for row in rd["R"]]
    return {"C": C, "R": rd["R"], "two_m": m2, "n": rd["n"]}


def kirchidx(A, tol=1e-9):
    """Kf = (1/2) sum_ij R_ij = n sum_{k>0} 1/lambda_k   (Klein & Randic 1993).

    The Kirchhoff index.  Both forms are returned because their
    agreement is the paper's identity and a useful check on the
    spectral computation.
    """
    rd = resdist(A, tol)
    n = rd["n"]
    kf_pairs = 0.5 * sum(sum(row) for row in rd["R"])
    lp = lappinv(A, tol)
    kf_spec = n * sum(1.0 / v for v in lp["eigenvalues"] if abs(v) > tol)
    return {"Kf": kf_pairs, "Kf_spectral": kf_spec, "n": n, "rank": lp["rank"]}


# ====================================================================
# Memoli, F. (2011), Gromov-Wasserstein distances and the metric
# approach to object matching, Found. Comput. Math. 11, 417-487.
# Solver: entropic projected gradient, Peyre, Cuturi & Solomon (2016),
# Gromov-Wasserstein averaging of kernel and distance matrices, ICML.
# ====================================================================


def gwdist(Cx, Cy, a, b, n_iter=50, epsilon=0.05, n_sinkhorn=50):
    """min_T sum |C^X_ij - C^Y_kl|^2 T_ik T_jl   (Memoli 2011).

    Gromov-Wasserstein discrepancy between two metric measure spaces
    given only their internal distance matrices -- no common ambient
    space is needed, which is the point of the construction.

    The objective is quartic and its exact minimisation is NP-hard, so
    the coupling is refined by ``n_iter`` fixed entropic projected
    gradient steps from the product coupling, each an inner Sinkhorn
    loop of ``n_sinkhorn`` fixed iterations.  Iteration counts are fixed
    rather than tolerance-driven, so the result is reproducible; the
    value at the product coupling is returned alongside so the
    improvement is visible.
    """
    X, Y = _as_mat(Cx), _as_mat(Cy)
    av, bv = _as_vec(a), _as_vec(b)
    n, m = len(av), len(bv)
    if len(X) != n or len(Y) != m:
        raise ValueError("Cx must be n x n and Cy must be m x m, matching a and b")
    if abs(sum(av) - 1.0) > 1e-9 or abs(sum(bv) - 1.0) > 1e-9:
        raise ValueError("a and b must each sum to 1")

    def cost(T):
        # sum_ijkl (X_ij - Y_kl)^2 T_ik T_jl
        tot = 0.0
        for i in range(n):
            for k in range(m):
                tik = T[i][k]
                if tik == 0.0:
                    continue
                for j in range(n):
                    d = X[i][j]
                    Tj = T[j]
                    for l in range(m):
                        e = d - Y[k][l]
                        tot += e * e * tik * Tj[l]
        return tot

    def grad(T):
        # dL/dT_ik = 2 sum_jl (X_ij - Y_kl)^2 T_jl
        G = [[0.0] * m for _ in range(n)]
        for i in range(n):
            for k in range(m):
                s = 0.0
                for j in range(n):
                    d = X[i][j]
                    Tj = T[j]
                    for l in range(m):
                        e = d - Y[k][l]
                        s += e * e * Tj[l]
                G[i][k] = 2.0 * s
        return G

    T = [[av[i] * bv[k] for k in range(m)] for i in range(n)]
    c0 = cost(T)
    for _ in range(n_iter):
        G = grad(T)
        K = [[math.exp(-G[i][k] / epsilon) * T[i][k] for k in range(m)] for i in range(n)]
        u = [1.0] * n
        v = [1.0] * m
        for _ in range(n_sinkhorn):
            for i in range(n):
                s = sum(K[i][k] * v[k] for k in range(m))
                u[i] = av[i] / s if s > 0 else 0.0
            for k in range(m):
                s = sum(K[i][k] * u[i] for i in range(n))
                v[k] = bv[k] / s if s > 0 else 0.0
        T = [[u[i] * K[i][k] * v[k] for k in range(m)] for i in range(n)]
    return {"T": T, "cost": cost(T), "cost_product": c0, "n_iter": int(n_iter), "n": n, "m": m}


# ====================================================================
# Palarea-Albaladejo, J. & Martin-Fernandez, J. A.
#   lrEM  : (2008), Computers & Geosciences 34, 902-917.
#   lrDA  : (2013) with Olea, Chemometrics & Intell. Lab. Syst. 126, 27-35.
# ====================================================================


def _alr(x):
    """Additive log-ratio against the last part; standard (Aitchison)."""
    d = len(x)
    return [math.log(x[i] / x[d - 1]) for i in range(d - 1)]


def _alr_inv(z, total):
    e = [math.exp(t) for t in z] + [1.0]
    s = sum(e)
    return [total * t / s for t in e]


def _lr_impute(X, dl, n_iter, draw=None):
    """Shared lrEM / lrDA loop.

    Below-detection entries are replaced by the conditional expectation
    of the alr-transformed part given the observed parts under a
    multivariate normal, truncated above at the alr of the detection
    limit.  lrDA adds a draw at each step instead of taking the
    expectation; ``draw[t][r]`` supplies the caller's standard normal
    variates so the augmentation is reproducible.
    """
    Xm = _as_mat(X)
    n, d = len(Xm), len(Xm[0])
    dlv = _as_vec(dl)
    if len(dlv) != d:
        raise ValueError("dl must give one detection limit per part")
    totals = [sum(r) for r in Xm]
    cens = [[Xm[i][j] < dlv[j] for j in range(d)] for i in range(n)]
    # start below-limit parts at 65% of the limit, the paper's initial fill
    W = [[(0.65 * dlv[j] if cens[i][j] else Xm[i][j]) for j in range(d)] for i in range(n)]
    for t in range(n_iter):
        Z = [_alr(row) for row in W]
        p = d - 1
        mu = [sum(Z[i][j] for i in range(n)) / n for j in range(p)]
        S = [[sum((Z[i][r] - mu[r]) * (Z[i][c] - mu[c]) for i in range(n)) / max(n - 1, 1) for c in range(p)] for r in range(p)]
        for i in range(n):
            for j in range(p):
                if not cens[i][j]:
                    continue
                obs = [c for c in range(p) if c != j]
                psi = math.log(dlv[j] / W[i][d - 1])
                if obs:
                    Soo = [[S[r][c] for c in obs] for r in obs]
                    sjo = [S[j][c] for c in obs]
                    w = solve(Soo, sjo, ridge=1e-10)
                    cm = mu[j] + sum(w[q] * (Z[i][obs[q]] - mu[obs[q]]) for q in range(len(obs)))
                    cv = S[j][j] - sum(w[q] * sjo[q] for q in range(len(obs)))
                else:
                    cm, cv = mu[j], S[j][j]
                sd = math.sqrt(max(cv, 1e-300))
                alpha = (psi - cm) / sd
                if draw is None:
                    # E[Z | Z < psi] for a truncated normal
                    denom = _Phi(alpha)
                    zj = cm - sd * (_phi(alpha) / denom) if denom > 1e-300 else psi
                else:
                    u = draw[t % len(draw)][i % len(draw[0])]
                    zj = min(cm + sd * u, psi)
                Z[i][j] = zj
                W[i] = _alr_inv(Z[i], totals[i])
    return {"X": W, "n": n, "n_parts": d, "n_iter": int(n_iter),
            "n_censored": sum(sum(1 for c in row if c) for row in cens)}


def lrem(X, dl, n_iter=20):
    """Log-ratio EM imputation of values below a detection limit.

    Palarea-Albaladejo & Martin-Fernandez (2008).  Replacing rounded
    zeros by a fraction of the detection limit distorts the covariance
    structure; lrEM instead imputes the conditional expectation under a
    normal model in alr coordinates, which preserves the ratios that
    compositional analysis actually uses.  ``n_iter`` is fixed.
    """
    return _lr_impute(X, dl, n_iter, draw=None)


def lrda(X, dl, draw, n_iter=20):
    """Log-ratio data augmentation for values below a detection limit.

    Palarea-Albaladejo, Martin-Fernandez & Olea (2013).  The Bayesian
    counterpart of :func:`lrem`: each step draws the censored part from
    its truncated conditional rather than taking the mean, so the
    imputation carries the right uncertainty.  Standard normal variates
    are supplied by the caller, so a run is reproducible.
    """
    if not draw:
        raise ValueError("lrda needs caller-supplied standard normal variates")
    return _lr_impute(X, dl, n_iter, draw=_as_mat(draw))


# ====================================================================
# Shao, J. & Wu, C. F. J. (1989), A general theory for jackknife
# variance estimation, Annals of Statistics 17, 1176-1197.
# ====================================================================


def jackd(theta, n, d):
    """Delete-d jackknife variance.

    v = (n-d) / (d * C(n,d)) * sum_s (theta_s - theta_bar)^2, the sum
    over the C(n,d) subsets of size n-d.  The delete-1 jackknife is
    inconsistent for non-smooth statistics such as the median; Shao &
    Wu show that deleting d > 1 restores consistency, and the leading
    factor is what keeps the estimator unbiased for the linear case.
    """
    tv = _as_vec(theta)
    n, d = int(n), int(d)
    if d < 1 or d >= n:
        raise ValueError("d must satisfy 1 <= d < n")
    nsub = math.comb(n, d)
    if len(tv) != nsub:
        raise ValueError(f"expected one estimate per size-(n-d) subset, i.e. {nsub}, got {len(tv)}")
    bar = _mean(tv)
    v = (n - d) / (d * nsub) * sum((t - bar) ** 2 for t in tv)
    return {"variance": v, "se": math.sqrt(max(v, 0.0)), "mean": bar,
            "n_subsets": nsub, "n": n, "d": d}


# ====================================================================
# Gibbons, J. D. & Chakraborti, S., Nonparametric Statistical Inference,
# Theorems 7.3.1 and 7.3.2 (moments of a linear rank statistic).
# ====================================================================


def lrankmom(a, m):
    """E(T_N) = m * abar,  Var(T_N) = m n / (N(N-1)) * sum (a_i - abar)^2.

    Moments of the linear rank statistic T_N = sum over the m
    treatment ranks of the scores a_i, when the m ranks are a simple
    random sample without replacement from the N = m + n scores.  Both
    theorems come out of sampling without replacement, which is why the
    variance carries the finite-population factor rather than being
    m * Var(a).
    """
    av = _as_vec(a)
    N = len(av)
    m = int(m)
    if not 0 < m < N:
        raise ValueError("m must satisfy 0 < m < N")
    n = N - m
    abar = _mean(av)
    ss = sum((t - abar) ** 2 for t in av)
    ev = m * abar
    var = m * n * ss / (N * (N - 1))
    return {"mean": ev, "variance": var, "se": math.sqrt(max(var, 0.0)),
            "score_mean": abar, "N": N, "m": m, "n": n}


# ====================================================================
# Cole, S. R. & Hernan, M. A. (2008), Constructing inverse probability
# weights for marginal structural models, Am. J. Epidemiol. 168, 656-664.
# ====================================================================


def wtrunc(w, q=0.99):
    """w_trunc = min(w, quantile_q(w)).

    Truncating inverse-probability weights trades a little bias for a
    large drop in variance: a single near-zero propensity produces a
    weight that dominates the estimate.  The quantile is the type-7
    (R default) sample quantile.
    """
    wv = _as_vec(w)
    if any(t < 0 for t in wv):
        raise ValueError("weights must be non-negative")
    if not 0 < float(q) <= 1:
        raise ValueError("q must lie in (0, 1]")
    s = sorted(wv)
    n = len(s)
    h = (n - 1) * float(q)
    lo = int(math.floor(h))
    hi = min(lo + 1, n - 1)
    cap = s[lo] + (h - lo) * (s[hi] - s[lo])
    out = [min(t, cap) for t in wv]
    return {"weights": out, "cap": cap, "n_truncated": sum(1 for t in wv if t > cap),
            "n": n, "mean_before": _mean(wv), "mean_after": _mean(out)}


# ====================================================================
# Yu, J. et al. (2006), A unified mixed-model method for association
# mapping that accounts for multiple levels of relatedness,
# Nature Genetics 38, 203-208.
# ====================================================================


def gwasmlm(y, X, snp, Vinv):
    """Per-SNP test of b = 0 in y = X mu + b snp + Z u + e.

    The unified mixed model absorbs population structure and kinship
    into the covariance of u; conditioning on it, the SNP effect is a
    generalised least squares coefficient.  ``Vinv`` is the inverse of
    the fitted total covariance, supplied by the caller so the variance
    components are estimated once rather than per SNP -- which is the
    computational point of the method.  The reported statistic is the
    Wald t on b.
    """
    yv, g = _as_vec(y), _as_vec(snp)
    Xm = _as_mat(X)
    Vi = _as_mat(Vinv)
    n = len(yv)
    if len(g) != n or len(Xm) != n or len(Vi) != n:
        raise ValueError("y, X, snp and Vinv must all have n rows")
    D = [Xm[i] + [g[i]] for i in range(n)]
    p = len(D[0])
    DtV = [[sum(D[i][r] * Vi[i][j] for i in range(n)) for j in range(n)] for r in range(p)]
    A = [[sum(DtV[r][j] * D[j][c] for j in range(n)) for c in range(p)] for r in range(p)]
    rhs = [sum(DtV[r][j] * yv[j] for j in range(n)) for r in range(p)]
    beta = solve(A, rhs)
    resid = [yv[i] - sum(D[i][c] * beta[c] for c in range(p)) for i in range(n)]
    rVr = sum(resid[i] * sum(Vi[i][j] * resid[j] for j in range(n)) for i in range(n))
    dfres = n - p
    if dfres <= 0:
        raise ValueError("no residual degrees of freedom")
    s2 = rVr / dfres
    Ainv_last = solve(A, [1.0 if c == p - 1 else 0.0 for c in range(p)])
    se = math.sqrt(max(s2 * Ainv_last[p - 1], 0.0))
    b = beta[p - 1]
    t = b / se if se > 0 else float("nan")
    return {"beta": b, "se": se, "statistic": t, "df": dfres,
            "coefficients": beta, "sigma2": s2, "n": n}


# ====================================================================
# Li, K. H., Meng, X.-L., Raghunathan, T. E. & Rubin, D. B. (1991),
# Significance levels from repeated p-values with multiply-imputed data,
# Statistica Sinica 1, 65-92.
# ====================================================================


def mitest(theta, U):
    """Multiply-imputed Wald test with the Li et al. (1991) reference df.

    Between-imputation variance B, within-imputation variance Ubar, and
    total T = Ubar + (1 + 1/m) B.  The statistic is referred to an F on
    (k, v) with the small-m denominator degrees of freedom of Li et al.,
    which is what stops a handful of imputations from producing a
    wildly optimistic p-value.  ``theta[i]`` is the k-vector estimate
    from imputation i and ``U[i]`` its k x k covariance.
    """
    Th = _as_mat(theta)
    m = len(Th)
    if m < 2:
        raise ValueError("need at least 2 imputations")
    k = len(Th[0])
    Us = [_as_mat(u) for u in U]
    qbar = [sum(Th[i][j] for i in range(m)) / m for j in range(k)]
    Ubar = [[sum(Us[i][r][c] for i in range(m)) / m for c in range(k)] for r in range(k)]
    B = [[sum((Th[i][r] - qbar[r]) * (Th[i][c] - qbar[c]) for i in range(m)) / (m - 1) for c in range(k)] for r in range(k)]
    # r1 = (1 + 1/m) tr(B Ubar^-1) / k
    BU = [solve(Ubar, [B[r][c] for r in range(k)]) for c in range(k)]
    tr = sum(BU[c][c] for c in range(k))
    r1 = (1.0 + 1.0 / m) * tr / k
    Tm = [[Ubar[r][c] * (1.0 + r1) for c in range(k)] for r in range(k)]
    sol = solve(Tm, qbar)
    D1 = sum(qbar[j] * sol[j] for j in range(k)) / k
    a = k * (m - 1)
    if a > 4:
        v = 4.0 + (a - 4.0) * (1.0 + (1.0 - 2.0 / a) / r1) ** 2
    else:
        v = 0.5 * a * (1.0 + 1.0 / k) * (1.0 + 1.0 / r1) ** 2
    return {"statistic": D1, "df1": k, "df2": v, "r": r1,
            "estimate": qbar, "m": m}


# ====================================================================
# Ge, T., Chen, C.-Y., Ni, Y., Feng, Y.-C. A. & Smoller, J. W. (2019),
# Polygenic prediction via Bayesian regression and continuous shrinkage
# priors, Nature Communications 10, 1776.
# ====================================================================


def csshrink(beta_hat, D, psi, n, sigma2=1.0):
    """Posterior mean of the SNP effects under a continuous shrinkage prior.

    beta_post = (D + Psi^-1)^-1 beta_hat, with D the LD matrix and
    Psi = diag(psi) the local shrinkage variances.  Continuous shrinkage
    is what separates PRS-CS from a fixed-threshold score: small effects
    are pulled hard toward zero while large ones are left nearly alone,
    so no p-value cutoff has to be chosen.  ``psi`` and ``sigma2`` are
    supplied rather than sampled, so the result is the conditional
    posterior mean and is reproducible.
    """
    bh = _as_vec(beta_hat)
    Dm = _as_mat(D)
    ps = _as_vec(psi)
    p = len(bh)
    if len(Dm) != p or len(ps) != p:
        raise ValueError("beta_hat, D and psi must be conformable")
    if any(t <= 0 for t in ps):
        raise ValueError("psi entries must be positive")
    nn = float(n)
    A = [[Dm[r][c] + (1.0 / ps[r] / nn if r == c else 0.0) for c in range(p)] for r in range(p)]
    post = solve(A, bh)
    shrink = [post[j] / bh[j] if bh[j] != 0 else float("nan") for j in range(p)]
    return {"beta": post, "shrinkage": shrink, "n": nn,
            "sigma2": float(sigma2), "n_snp": p}


# ====================================================================
# He, X. et al. (2020), Temporal dynamics in viral shedding and
# transmissibility of COVID-19, Nature Medicine 26, 672-675.
# ====================================================================


def shedcurve(days, load, t_peak, t_plateau):
    """Piecewise log-linear shedding curve: rise, plateau, decay.

    Viral load is fitted on the log10 scale in three segments split at
    ``t_peak`` and ``t_plateau``: a rising slope before the peak, a flat
    level between, and a decay slope after.  Each segment is an ordinary
    least squares fit, so the result is closed form.  The shape matters
    clinically because peak shedding precedes symptom onset, which is
    the paper's finding.
    """
    d, v = _as_vec(days), _as_vec(load)
    if len(d) != len(v):
        raise ValueError("days and load must have the same length")
    if any(t <= 0 for t in v):
        raise ValueError("viral load must be positive to take log10")
    y = [math.log10(t) for t in v]
    tp, tq = float(t_peak), float(t_plateau)
    if not tp < tq:
        raise ValueError("t_peak must be strictly before t_plateau")

    def _slope(idx):
        if len(idx) < 2:
            return float("nan"), (_mean([y[i] for i in idx]) if idx else float("nan"))
        xs = [d[i] for i in idx]
        ys = [y[i] for i in idx]
        mx, my = _mean(xs), _mean(ys)
        sxx = sum((t - mx) ** 2 for t in xs)
        if sxx == 0:
            return float("nan"), my
        b = sum((xs[q] - mx) * (ys[q] - my) for q in range(len(xs))) / sxx
        return b, my - b * mx

    rise = [i for i in range(len(d)) if d[i] < tp]
    plat = [i for i in range(len(d)) if tp <= d[i] <= tq]
    dec = [i for i in range(len(d)) if d[i] > tq]
    br, ar = _slope(rise)
    bd, ad = _slope(dec)
    return {
        "rise_slope": br, "rise_intercept": ar,
        "plateau_level": _mean([y[i] for i in plat]) if plat else float("nan"),
        "decay_slope": bd, "decay_intercept": ad,
        "peak_load": max(y), "peak_day": d[max(range(len(y)), key=lambda i: y[i])],
        "n": len(d), "n_rise": len(rise), "n_plateau": len(plat), "n_decay": len(dec),
    }


# ====================================================================
# Targeted learning.
#   CV-TMLE : Zheng, W. & van der Laan, M. J. (2011), Cross-validated
#             targeted minimum-loss-based estimation, in Targeted
#             Learning, Springer, ch. 27.
#   Natural direct / indirect effects : Zheng, W. & van der Laan, M. J.
#             (2012), Targeted maximum likelihood estimation of natural
#             direct effects, Int. J. Biostatistics 8(1), art. 3.
# ====================================================================


def cvtmle(y, a, q0, q1, g, fold, n_newton=50):
    """Cross-validated TMLE of the ATE.

    Within each fold the initial fit is the one trained on the other
    folds, so the targeting step never sees the data it is evaluated
    on: that is what removes the empirical-process condition and lets
    data-adaptive nuisance fits be used honestly.  The fluctuation
    parameter solves the score equation for the clever covariate
    H = A/g - (1-A)/(1-g) by a fixed number of Newton steps.

    ``y`` must lie in [0, 1] (bounded outcomes, or already rescaled).
    """
    yv, av, g0, g1, gv = _as_vec(y), _as_vec(a), _as_vec(q0), _as_vec(q1), _as_vec(g)
    fv = [int(t) for t in fold]
    n = len(yv)
    if not (len(av) == len(g0) == len(g1) == len(gv) == len(fv) == n):
        raise ValueError("all inputs must have the same length")
    if any(t < 0.0 or t > 1.0 for t in yv):
        raise ValueError("y must be bounded in [0, 1]")
    if any(t <= 0.0 or t >= 1.0 for t in gv):
        raise ValueError("propensities must lie strictly inside (0, 1)")

    def lg(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return math.log(p / (1 - p))

    psi_fold, eps_fold = [], []
    for f in sorted(set(fv)):
        idx = [i for i in range(n) if fv[i] == f]
        H = [av[i] / gv[i] - (1 - av[i]) / (1 - gv[i]) for i in idx]
        Qa = [g1[i] if av[i] == 1 else g0[i] for i in idx]
        eps = 0.0
        for _ in range(n_newton):
            s = sc = 0.0
            for q in range(len(idx)):
                p = 1.0 / (1.0 + math.exp(-(lg(Qa[q]) + eps * H[q])))
                s += H[q] * (yv[idx[q]] - p)
                sc -= H[q] * H[q] * p * (1 - p)
            if sc == 0.0:
                break
            eps -= s / sc
        q1s = [1.0 / (1.0 + math.exp(-(lg(g1[i]) + eps * (1.0 / gv[i])))) for i in idx]
        q0s = [1.0 / (1.0 + math.exp(-(lg(g0[i]) - eps * (1.0 / (1 - gv[i]))))) for i in idx]
        psi_fold.append(sum(q1s[q] - q0s[q] for q in range(len(idx))) / len(idx))
        eps_fold.append(eps)
    psi = sum(psi_fold) / len(psi_fold)
    ic = []
    for i in range(n):
        H = av[i] / gv[i] - (1 - av[i]) / (1 - gv[i])
        Qa = g1[i] if av[i] == 1 else g0[i]
        ic.append(H * (yv[i] - Qa) + (g1[i] - g0[i]) - psi)
    se = math.sqrt(sum(t * t for t in ic) / n / n)
    return {"estimate": psi, "se": se, "psi_fold": psi_fold, "eps_fold": eps_fold,
            "n_folds": len(psi_fold), "n": n}


def ndeff(y10, y00):
    """Natural direct effect: E[Y(1, M(0))] - E[Y(0, M(0))].

    The effect of treatment holding the mediator at the distribution it
    would have taken under control -- the path that does not run through
    the mediator.  Both cross-world quantities are supplied by the
    caller because neither is identified from data without an
    assumption; this routine contrasts them and does not assert one.
    """
    a, b = _as_vec(y10), _as_vec(y00)
    if len(a) != len(b):
        raise ValueError("y10 and y00 must have the same length")
    d = [a[i] - b[i] for i in range(len(a))]
    n = len(d)
    se = math.sqrt(_var(d) / n) if n > 1 else float("nan")
    return {"estimate": _mean(d), "se": se, "mean_y10": _mean(a),
            "mean_y00": _mean(b), "n": n}


def nieff(y11, y10):
    """Natural indirect effect: E[Y(1, M(1))] - E[Y(1, M(0))].

    The effect that runs through the mediator, holding treatment fixed
    at 1 and moving only the mediator's distribution.  Adding this to
    the natural direct effect recovers the total effect exactly, which
    is the decomposition the pair exists for.
    """
    a, b = _as_vec(y11), _as_vec(y10)
    if len(a) != len(b):
        raise ValueError("y11 and y10 must have the same length")
    d = [a[i] - b[i] for i in range(len(a))]
    n = len(d)
    se = math.sqrt(_var(d) / n) if n > 1 else float("nan")
    return {"estimate": _mean(d), "se": se, "mean_y11": _mean(a),
            "mean_y10": _mean(b), "n": n}


# ====================================================================
# Kunzel, S. R., Sekhon, J. S., Bickel, P. J. & Yu, B. (2019),
# Metalearners for estimating heterogeneous treatment effects using
# machine learning, PNAS 116, 4156-4165.
# ====================================================================


def xlearn(tau1, tau0, g):
    """X-learner combination: tau = g tau0 + (1 - g) tau1.

    The X-learner imputes each unit's missing arm from the other arm's
    fitted response, then blends the two resulting effect estimates by
    the propensity.  The weighting is the point: when one arm is much
    smaller, its own imputed effect is noisy, and weighting by g pushes
    the estimate toward the estimate built from the larger arm.
    """
    t1, t0, gv = _as_vec(tau1), _as_vec(tau0), _as_vec(g)
    if not (len(t1) == len(t0) == len(gv)):
        raise ValueError("tau1, tau0 and g must have the same length")
    if any(t < 0 or t > 1 for t in gv):
        raise ValueError("propensities must lie in [0, 1]")
    tau = [gv[i] * t0[i] + (1 - gv[i]) * t1[i] for i in range(len(gv))]
    n = len(tau)
    se = math.sqrt(_var(tau) / n) if n > 1 else float("nan")
    return {"tau": tau, "ate": _mean(tau), "se": se, "n": n}


# ====================================================================
# Deterministic components of published neural architectures.  Each of
# these is a closed-form operation defined in its paper, evaluated on
# caller-supplied weights: nothing here is trained.
# ====================================================================


def rope(q, m, theta):
    """Rotary position embedding: f(q, m) = R_{theta, m} q.

    Su et al. (2021), RoFormer.  Coordinate pairs (q_{2i}, q_{2i+1}) are
    rotated by angle m * theta_i.  Because the rotation is orthogonal,
    the inner product of two rotated vectors depends only on their
    relative position m - n, which is the property the construction was
    built to get.
    """
    qv, th = _as_vec(q), _as_vec(theta)
    if len(qv) % 2 != 0:
        raise ValueError("q must have even length (rotations act on pairs)")
    half = len(qv) // 2
    if len(th) != half:
        raise ValueError("theta must give one angle per coordinate pair")
    mm = float(m)
    out = [0.0] * len(qv)
    for i in range(half):
        c, s = math.cos(mm * th[i]), math.sin(mm * th[i])
        a, b = qv[2 * i], qv[2 * i + 1]
        out[2 * i] = c * a - s * b
        out[2 * i + 1] = s * a + c * b
    return {"q": out, "m": mm, "norm": math.sqrt(sum(t * t for t in out)), "n": len(qv)}


def grpnorm(x, n_groups, eps=1e-5):
    """Group normalisation over channel groups.

    Wu & He (2018).  Channels are split into ``n_groups`` groups and each
    group is standardised over its own elements.  Unlike batch norm the
    statistics come from one sample, so the result does not depend on
    batch size -- which is why it holds up at batch size 1.
    ``x`` is a flat channel-major vector of length C * S.
    """
    xv = _as_vec(x)
    G = int(n_groups)
    if G < 1 or len(xv) % G != 0:
        raise ValueError("length of x must be divisible by n_groups")
    per = len(xv) // G
    out = [0.0] * len(xv)
    mus, sds = [], []
    for g in range(G):
        seg = xv[g * per:(g + 1) * per]
        mu = _mean(seg)
        var = sum((t - mu) ** 2 for t in seg) / per
        sd = math.sqrt(var + float(eps))
        mus.append(mu)
        sds.append(sd)
        for j in range(per):
            out[g * per + j] = (seg[j] - mu) / sd
    return {"x": out, "mean": mus, "sd": sds, "n_groups": G, "group_size": per}


def sumpl(H):
    """Graph readout by summing node embeddings: h_G = sum_v h_v.

    Sum pooling is the readout that makes a message-passing network as
    discriminative as the Weisfeiler-Lehman test (Xu et al. 2019, GIN);
    mean and max pooling both collapse multisets that sum pooling keeps
    apart, so all three are returned for comparison.
    """
    Hm = _as_mat(H)
    if not Hm:
        raise ValueError("H must have at least one node")
    d = len(Hm[0])
    s = [sum(row[j] for row in Hm) for j in range(d)]
    return {"sum": s, "mean": [t / len(Hm) for t in s],
            "max": [max(row[j] for row in Hm) for j in range(d)],
            "n_nodes": len(Hm), "dim": d}


def ginagg(A, H, eps=0.0):
    """GIN aggregation: h_v <- (1 + eps) h_v + sum_{u in N(v)} h_u.

    Xu et al. (2019), Graph Isomorphism Network.  The (1 + eps) factor
    on the centre node is what keeps the self-representation
    distinguishable from the neighbour sum, so the aggregator is
    injective on multisets.  The learned MLP that follows is left to the
    caller; this is the aggregation step itself.
    """
    Am, Hm = _as_mat(A), _as_mat(H)
    n = len(Hm)
    if len(Am) != n or any(len(r) != n for r in Am):
        raise ValueError("A must be n x n matching H")
    e = float(eps)
    out = [[(1.0 + e) * Hm[v][j] + sum(Am[v][u] * Hm[u][j] for u in range(n)) for j in range(len(Hm[0]))] for v in range(n)]
    return {"H": out, "eps": e, "n_nodes": n, "dim": len(Hm[0])}


def _sym_norm(A, self_loops):
    n = len(A)
    M = [[A[i][j] + (1.0 if (self_loops and i == j) else 0.0) for j in range(n)] for i in range(n)]
    deg = [sum(M[i]) for i in range(n)]
    for i in range(n):
        if deg[i] <= 0:
            raise ValueError(f"node {i} has non-positive degree; cannot normalise")
    return [[M[i][j] / math.sqrt(deg[i] * deg[j]) for j in range(n)] for i in range(n)]


def sgcprop(A, X, K):
    """Simplified graph convolution: S^K X, S = D^-1/2 (A + I) D^-1/2.

    Wu et al. (2019).  Collapsing the nonlinearities between graph
    convolution layers leaves a fixed linear smoothing operator applied
    K times, which can be precomputed once; the classifier that follows
    is then an ordinary logistic regression.  The result is the
    propagated feature matrix.
    """
    S = _sym_norm(_as_mat(A), True)
    out = _as_mat(X)
    for _ in range(int(K)):
        out = matmul(S, out)
    return {"X": out, "K": int(K), "n_nodes": len(S), "dim": len(out[0])}


def lgcnprop(A, E, K, alpha=None):
    """LightGCN: e = sum_k alpha_k S^k e, S = D^-1/2 A D^-1/2.

    He et al. (2020).  Feature transformation and nonlinearity are
    dropped entirely -- only neighbourhood averaging remains -- and the
    layer outputs are combined by fixed weights, uniform 1/(K+1) by
    default as in the paper.
    """
    S = _sym_norm(_as_mat(A), False)
    K = int(K)
    w = [1.0 / (K + 1)] * (K + 1) if alpha is None else _as_vec(alpha)
    if len(w) != K + 1:
        raise ValueError("alpha must give one weight per layer including layer 0")
    cur = _as_mat(E)
    acc = [[w[0] * v for v in row] for row in cur]
    for k in range(1, K + 1):
        cur = matmul(S, cur)
        for i in range(len(acc)):
            for j in range(len(acc[0])):
                acc[i][j] += w[k] * cur[i][j]
    return {"E": acc, "K": K, "alpha": w, "n_nodes": len(acc), "dim": len(acc[0])}


def linucb(x, theta, Ainv, alpha=1.0):
    """LinUCB arm scores: p_a = theta_a' x + alpha sqrt(x' A_a^-1 x).

    Li et al. (2010).  The bonus term is a confidence radius, not noise:
    it is large exactly for arms whose design matrix has seen little
    variation along x, so exploration is directed rather than random.
    ``theta[a]`` and ``Ainv[a]`` are the per-arm parameter and inverse
    design matrix.
    """
    xv = _as_vec(x)
    Th = _as_mat(theta)
    d = len(xv)
    if any(len(r) != d for r in Th):
        raise ValueError("each theta must match the context dimension")
    if len(Ainv) != len(Th):
        raise ValueError("need one inverse design matrix per arm")
    mean, bonus, score = [], [], []
    for a in range(len(Th)):
        Ai = _as_mat(Ainv[a])
        if len(Ai) != d or any(len(r) != d for r in Ai):
            raise ValueError("each Ainv must be d x d")
        q = sum(xv[i] * sum(Ai[i][j] * xv[j] for j in range(d)) for i in range(d))
        if q < 0:
            raise ValueError("x' Ainv x is negative; Ainv must be positive semidefinite")
        mu = sum(Th[a][i] * xv[i] for i in range(d))
        bo = float(alpha) * math.sqrt(q)
        mean.append(mu)
        bonus.append(bo)
        score.append(mu + bo)
    best = max(range(len(score)), key=lambda a: score[a])
    return {"score": score, "mean": mean, "bonus": bonus, "arm": best,
            "n_arms": len(score), "alpha": float(alpha)}


def ssmk(A, B, C, L):
    """State-space kernel: K_l = C A^l B, y = K * x.

    Gu, Goel & Re (2022), S4.  A linear state-space model unrolled in
    time is a convolution with the kernel (CB, CAB, CA^2 B, ...), so a
    recurrence of length L becomes one convolution -- that equivalence
    is what makes the model trainable at long sequence lengths.
    """
    Am, Bv, Cv = _as_mat(A), _as_vec(B), _as_vec(C)
    n = len(Am)
    if len(Bv) != n or len(Cv) != n:
        raise ValueError("B and C must match the state dimension of A")
    K = []
    v = list(Bv)
    for _ in range(int(L)):
        K.append(sum(Cv[i] * v[i] for i in range(n)))
        v = matvec(Am, v)
    return {"K": K, "L": int(L), "state_dim": n}


def ssmconv(K, x):
    """Causal convolution y_t = sum_{l<=t} K_l x_{t-l}; standard."""
    Kv, xv = _as_vec(K), _as_vec(x)
    return [sum(Kv[l] * xv[t - l] for l in range(min(t + 1, len(Kv)))) for t in range(len(xv))]


def fftperiod(x, k=1):
    """Dominant periods from the amplitude spectrum.

    Wu et al. (2023), TimesNet.  The periods that carry the most
    amplitude are read off the discrete Fourier transform and used to
    fold the 1-D series into a 2-D tensor, which is how intraperiod and
    interperiod variation get separated.  Frequency 0 is excluded and
    only the first half of the spectrum is used, since the rest is its
    mirror image.
    """
    xv = _as_vec(x)
    n = len(xv)
    if n < 4:
        raise ValueError("need at least 4 observations")
    amps = dft_amp(xv)
    half = n // 2
    cand = list(range(1, half + 1))
    kk = int(k)
    if kk < 1 or kk > len(cand):
        raise ValueError(f"k must lie in 1..{len(cand)}")
    # ties broken by the lower frequency, so the ordering is total
    order = sorted(cand, key=lambda f: (-amps[f], f))[:kk]
    return {"frequency": order, "period": [n / f for f in order],
            "amplitude": [amps[f] for f in order], "spectrum": amps[:half + 1], "n": n}


def serdecomp(x, kernel):
    """Series decomposition and autocorrelation.

    Wu et al. (2021), Autoformer.  A moving average of odd width
    ``kernel``, replicate-padded at both ends, is the trend; the
    remainder is the seasonal part.  The autocorrelation of the
    seasonal part is what the architecture's auto-correlation block
    scores periods with, so it is returned alongside.
    """
    xv = _as_vec(x)
    n, kk = len(xv), int(kernel)
    if kk < 1 or kk % 2 == 0:
        raise ValueError("kernel must be a positive odd integer")
    if kk > n:
        raise ValueError("kernel must not exceed the series length")
    h = kk // 2
    pad = [xv[0]] * h + xv + [xv[-1]] * h
    trend = [sum(pad[i:i + kk]) / kk for i in range(n)]
    seas = [xv[i] - trend[i] for i in range(n)]
    m = _mean(seas)
    den = sum((t - m) ** 2 for t in seas)
    acf = [(sum((seas[i] - m) * (seas[i + lag] - m) for i in range(n - lag)) / den) if den > 0 else float("nan")
           for lag in range(n)]
    return {"trend": trend, "seasonal": seas, "acf": acf, "kernel": kk, "n": n}
