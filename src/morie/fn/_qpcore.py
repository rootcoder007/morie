"""Dense linear algebra and the Goldfarb-Idnani dual active-set QP solver.

Goldfarb, D. and Idnani, A. (1983). A numerically stable dual method for solving strictly
convex quadratic programs. Mathematical Programming 27, 1-33.
"""

import math


def ssum(it):
    # plain left-to-right summation (sum() of floats is compensated from Python 3.12 on)
    s = 0.0
    for v in it:
        s += v
    return s


def dot(a, b):
    s = 0.0
    for u, v in zip(a, b):
        s += u * v
    return s


def matvec(M, v):
    return [dot(r, v) for r in M]


def solve(A, b):
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-300:
            raise ZeroDivisionError("singular system")
        for r in range(c + 1, n):
            t = M[r][c] / M[c][c]
            if t != 0.0:
                for k in range(c, n + 1):
                    M[r][k] -= t * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n]
        for k in range(r + 1, n):
            s -= M[r][k] * x[k]
        x[r] = s / M[r][r]
    return x


def inverse(A):
    n = len(A)
    cols = [solve(A, [float(i == j) for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def goldfarb_idnani(G, a, C, b, meq=0, tol=1e-12, max_iter=None):
    """min 1/2 x'Gx + a'x subject to c_k'x = b_k (k < meq) and c_k'x >= b_k (k >= meq).

    G must be symmetric positive definite; C is a list of constraint vectors c_k.
    Dual active set (Goldfarb and Idnani 1983): start from the unconstrained minimiser,
    repeatedly add the first violated equality or the most violated inequality, taking
    full or partial steps and dropping blocking inequalities, until every constraint
    holds. With the active normals N, H = G^-1 - G^-1 N (N'G^-1 N)^-1 N'G^-1 and
    N* = (N'G^-1 N)^-1 N'G^-1 are recomputed after each change of the active set.
    Returns (x, multipliers u per constraint, active index list); raises ValueError when
    the constraints are inconsistent.
    """
    n, m = len(a), len(C)
    Gi = inverse(G)
    x = [-v for v in matvec(Gi, a)]
    A, uA, sA = [], [], []
    max_iter = 50 * (n + m) + 50 if max_iter is None else max_iter

    def build():
        if not A:
            return [list(r) for r in Gi], []
        N = [[sg * v for v in C[k]] for k, sg in zip(A, sA)]
        GiN = [matvec(Gi, nk) for nk in N]  # columns G^-1 n_k
        M = [[dot(N[i], GiN[j]) for j in range(len(A))] for i in range(len(A))]
        Mi = inverse(M)
        Ns = [[ssum(Mi[i][q] * GiN[q][c] for q in range(len(A))) for c in range(n)] for i in range(len(A))]
        H = [[Gi[r][c] - ssum(GiN[i][r] * Ns[i][c] for i in range(len(A))) for c in range(n)] for r in range(n)]
        return H, Ns

    H, Ns = build()
    for _ in range(max_iter):
        slack = [dot(C[k], x) - b[k] for k in range(m)]
        p, sp = None, 0.0
        for k in range(meq):
            if k not in A and abs(slack[k]) > tol * max(1.0, abs(b[k])):
                p, sp = k, slack[k]
                break
        if p is None:
            worst = -tol
            for k in range(meq, m):
                if k not in A and slack[k] < worst * max(1.0, abs(b[k])):
                    worst, p, sp = slack[k], k, slack[k]
        if p is None:
            u = [0.0] * m
            for k, v, sg in zip(A, uA, sA):
                u[k] = sg * v
            return x, u, sorted(A)
        sgn = 1.0 if (p >= meq or sp < 0) else -1.0  # an equality is approached from its violated side
        npv = [sgn * v for v in C[p]]
        up = 0.0
        while True:
            z = matvec(H, npv)
            r = matvec(Ns, npv) if A else []
            t1, kdrop = math.inf, -1
            for i, k in enumerate(A):
                if k >= meq and r[i] > 1e-14 and uA[i] / r[i] < t1:
                    t1, kdrop = uA[i] / r[i], i
            zn = dot(z, npv)
            # z = 0 when n_p depends on the active normals; test it relative to n'G^-1 n
            t2 = -(sgn * (dot(C[p], x) - b[p])) / zn if zn > 1e-11 * dot(npv, matvec(Gi, npv)) else math.inf
            t = min(t1, t2)
            if t == math.inf:
                raise ValueError("the constraints are inconsistent")
            if t2 < math.inf:
                x = [xi + t * zi for xi, zi in zip(x, z)]
            uA = [ui - t * ri for ui, ri in zip(uA, r)]
            up += t
            if t == t2:
                A.append(p)
                uA.append(up)
                sA.append(sgn)
                H, Ns = build()
                break
            del A[kdrop]
            del uA[kdrop]
            del sA[kdrop]
            H, Ns = build()
    raise RuntimeError("Goldfarb-Idnani did not terminate")


def simplex_standard(c, A, b, tol=1e-10, max_iter=5000):
    """min c'x s.t. Ax = b, x >= 0 by the two-phase tableau simplex with Bland's rule.

    Dantzig, G. B. (1963). Linear Programming and Extensions; Bland, R. G. (1977). New finite
    pivoting rules for the simplex method. Mathematics of Operations Research 2, 103-107.
    Returns (x, status) with status "optimal", "infeasible" or "unbounded".
    """
    m, n = len(A), len(c)
    T = []
    for i in range(m):
        sg = -1.0 if b[i] < 0 else 1.0
        T.append([sg * float(v) for v in A[i]] + [1.0 if k == i else 0.0 for k in range(m)] + [sg * float(b[i])])
    basis = [n + i for i in range(m)]
    W = n + m

    def pivot(r, q):
        pv = T[r][q]
        T[r] = [v / pv for v in T[r]]
        for i in range(m):
            if i != r and T[i][q] != 0.0:
                f = T[i][q]
                T[i] = [a - f * b_ for a, b_ in zip(T[i], T[r])]
        basis[r] = q

    def run(cost, allowed):
        for _ in range(max_iter):
            red = []
            for j in range(W):
                if j in allowed:
                    z = cost[j]
                    for i in range(m):
                        z -= cost[basis[i]] * T[i][j]
                    red.append((j, z))
            q = next((j for j, z in red if z < -tol and j not in basis), None)
            if q is None:
                return "optimal"
            ratios = [(T[i][W] / T[i][q], basis[i], i) for i in range(m) if T[i][q] > tol]
            if not ratios:
                return "unbounded"
            best = min(r[0] for r in ratios)
            r = min((bi, i) for t, bi, i in ratios if t <= best + tol)[1]
            pivot(r, q)
        raise RuntimeError("simplex did not terminate")

    run([0.0] * n + [1.0] * m, set(range(W)))
    if ssum(T[i][W] for i in range(m) if basis[i] >= n) > 1e-8 * max(1.0, ssum(abs(v) for v in b)):
        return None, "infeasible"
    for i in range(m):  # drive remaining artificials out of the basis
        if basis[i] >= n:
            q = next((j for j in range(n) if abs(T[i][j]) > tol), None)
            if q is not None:
                pivot(i, q)
    status = run([float(v) for v in c] + [0.0] * m, set(range(n)))
    x = [0.0] * n
    for i in range(m):
        if basis[i] < n:
            x[basis[i]] = T[i][W]
    return x, status
