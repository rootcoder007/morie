# morie.fn -- internal helper (rootcoder007/morie)
"""Barrier path-following solver for small semidefinite programs.

Shared by :mod:`morie.fn.cvxsdp` (the SDP itself) and
:mod:`morie.fn.cvxqsv` (the SDP relaxation of a QCQP), which is the only
reason this is a module rather than a private function.

ponytail: dense Newton on the log-det barrier, no sparsity exploitation
and no primal-dual predictor-corrector. Fine to a few dozen variables
and matrix blocks in the low hundreds; past that, hand the problem to a
real conic solver.
"""

from __future__ import annotations

from . import _array_core as np

__all__ = ["solve_sdp"]


def _F(F0, Fs, x):
    M = F0.copy()
    for xi, Fi in zip(x, Fs):
        M = M + xi * Fi
    return M


def _min_eig(F0, Fs, x):
    return float(np.linalg.eigvalsh(_F(F0, Fs, x))[0])


def _strictly_feasible(F0, Fs, x0, margin):
    """Phase I: any x with F(x) > 0, or None if none was found."""
    n = len(Fs)
    from ._sci_core import minimize

    starts = []
    if x0 is not None:
        starts.append(np.atleast_1d(np.asarray(x0, dtype=float)).ravel())
    starts.append(np.zeros(n))
    starts.append(np.ones(n))
    for s in starts:
        if s.size == n and _min_eig(F0, Fs, s) > margin:
            return s
    # Maximising the smallest eigenvalue is the standard phase-I
    # problem. It is nonsmooth at eigenvalue crossings, so a
    # derivative-free method is the right tool at these sizes -- but it
    # is also UNBOUNDED whenever some F_i is positive definite (scale x
    # up and lambda_min grows without limit), so the search has to be
    # stopped the moment it lands anywhere strictly feasible rather than
    # left to run to an "optimum" at 1e308.
    target = max(margin, 1e-06 * max(1.0, float(np.abs(F0).max())))
    found = []

    class _Feasible(Exception):
        pass

    def probe(z):
        val = _min_eig(F0, Fs, z)
        if np.isfinite(val) and val > target:
            found.append(np.array(z, dtype=float))
            raise _Feasible
        return -val if np.isfinite(val) else 1e100

    for s in starts:
        if s.size != n:
            continue
        try:
            minimize(probe, s, method="Nelder-Mead",
                     options={"maxiter": 4000, "xatol": 1e-10,
                              "fatol": 1e-12})
        except _Feasible:
            return found[-1]
    return None


def solve_sdp(c, F0, Fs, x0=None, t0=1.0, mu=15.0, tol=1e-09,
              max_outer=80, max_newton=80, feas_margin=1e-09):
    r"""Minimise ``c'x`` subject to ``F0 + sum x_i F_i >= 0`` (Loewner).

    Returns ``(x, info)`` with ``info`` carrying the attained objective,
    the final slack matrix, its eigenvalues, the barrier suboptimality
    bound ``m / t``, and whether phase I and the outer loop succeeded.
    """
    c = np.atleast_1d(np.asarray(c, dtype=float)).ravel()
    F0 = np.atleast_2d(np.asarray(F0, dtype=float))
    Fs = [np.atleast_2d(np.asarray(Fi, dtype=float)) for Fi in Fs]
    n = c.size
    if len(Fs) != n:
        raise ValueError(f"c has {n} entries but {len(Fs)} matrices were given")
    m = F0.shape[0]
    if F0.shape[0] != F0.shape[1]:
        raise ValueError("F0 must be square")
    for k, Fi in enumerate(Fs):
        if Fi.shape != F0.shape:
            raise ValueError(f"F[{k}] has shape {Fi.shape}, expected {F0.shape}")
    F0 = 0.5 * (F0 + F0.T)
    Fs = [0.5 * (Fi + Fi.T) for Fi in Fs]

    x = _strictly_feasible(F0, Fs, x0, feas_margin)
    if x is None:
        return None, {"feasible": False, "phase1": False,
                      "message": "no strictly feasible point found; the "
                                 "constraint set is empty or has empty "
                                 "interior (Slater fails)"}

    t = float(t0)
    converged = False
    for _ in range(int(max_outer)):
        for _ in range(int(max_newton)):
            M = _F(F0, Fs, x)
            Minv = np.linalg.inv(M)
            # d/dx_i [-logdet F] = -tr(F^-1 F_i);
            # d2/dx_i dx_j       =  tr(F^-1 F_i F^-1 F_j).
            MF = [Minv @ Fi for Fi in Fs]
            grad = t * c - np.array([np.trace(P) for P in MF])
            H = np.array([[np.trace(MF[i] @ MF[j]) for j in range(n)]
                          for i in range(n)])
            try:
                step = -np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(H, grad, rcond=None)[0]
            lam2 = float(-grad @ step)
            if not np.isfinite(lam2) or lam2 / 2.0 <= 1e-12:
                break

            def phi(z):
                Mz = _F(F0, Fs, z)
                ev = np.linalg.eigvalsh(Mz)
                if ev[0] <= 0:
                    return np.inf
                return float(t * (c @ z) - np.sum(np.log(ev)))

            f_now = phi(x)
            s = 1.0
            while s > 1e-14:
                trial = x + s * step
                if phi(trial) <= f_now - 0.25 * s * lam2:
                    break
                s *= 0.5
            if s <= 1e-14:
                break
            x = x + s * step
        if m / t < tol:
            converged = True
            break
        t *= mu
    M = _F(F0, Fs, x)
    ev = np.linalg.eigvalsh(M)
    return x, {
        "objective": float(c @ x), "slack": M, "eigenvalues": ev,
        "gap_bound": float(m / t), "feasible": bool(ev[0] > -1e-08),
        "phase1": True, "converged": bool(converged), "t": float(t),
    }
