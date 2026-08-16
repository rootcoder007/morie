"""GP hyperparameters by MCMC, with the latent function integrated out.

Optimising a Gaussian-process kernel's hyperparameters and then
predicting as if those values were known throws away the part of the
uncertainty that usually matters most. The lengthscale is not well
determined by a few dozen points, and a predictive interval computed at
its maximiser is narrower than the data supports. The fix is to SAMPLE
the hyperparameters and average the predictions over the sample.

Doing that is easy for a Gaussian likelihood and awkward otherwise, and
the awkwardness is the subject of Murray and Adams (2010): the latent
function f and the hyperparameters theta are strongly coupled a
posteriori, because theta controls the prior covariance of f. A sampler
that moves theta while holding f fixed proposes a function that is
wildly improbable under the new kernel, and every such move is rejected.
Their answer is to reparameterise so the coupling is broken. Three
routes are implemented, and they target the SAME posterior -- which is
what the parity harness anchors on.

  "marginal"   For a Gaussian likelihood, f can be integrated out in
               closed form: y ~ N(0, K_theta + s2n I). Then there is no
               latent variable to couple to and theta is sampled from
               its exact marginal posterior. Correct, cheapest, and the
               reference the other two are checked against.

  "whitened"   Murray and Adams' ancillary augmentation. Write
               f = L_theta nu with L_theta the Cholesky factor and
               nu ~ N(0, I). Now nu carries no dependence on theta, so
               moving theta with nu fixed drags f along coherently
               instead of stranding it. nu itself is updated by
               elliptical slice sampling, which needs no step size
               because its proposal is an exact ellipse through the
               prior.

  "surrogate"  Their surrogate data method. Draw g ~ N(f, S), treat g
               as fixed, and reparameterise f around the conditional
               mean and covariance it induces. The surrogate acts as a
               noisy summary of f that theta must remain consistent
               with, which is a weaker tie than f itself and so allows
               larger moves in theta. S is taken as the likelihood's
               own noise, which is the paper's default choice.

Hyperparameters are slice sampled one coordinate at a time on the log
scale (Neal 2003: stepping out, then shrinkage). Slice sampling is used
rather than Metropolis because it has no step size to tune -- the
stepping-out procedure finds the scale itself, which matters when the
same code has to work for a lengthscale and a noise variance without
being retuned.

Kernels, all selectable: squared exponential, Matern 3/2, Matern 5/2.
The Materns are there because the squared exponential assumes a function
that is infinitely differentiable, which is a very strong statement
about a real process and is almost never checked.

References
  Murray, I. and Adams, R.P. (2010) "Slice sampling covariance
    hyperparameters of latent Gaussian models." Advances in Neural
    Information Processing Systems 23, 1732-1740. The whitened
    (ancillary) and surrogate data reparameterisations.
  Murray, I., Adams, R.P. and MacKay, D.J.C. (2010) "Elliptical slice
    sampling." AISTATS 9, 541-548. The latent update.
  Neal, R.M. (2003) "Slice sampling." Annals of Statistics 31(3),
    705-767. Stepping out and shrinkage.
  Rasmussen, C.E. and Williams, C.K.I. (2006) "Gaussian Processes for
    Machine Learning." MIT Press. Chapter 2: the marginal likelihood and
    the predictive equations; chapter 4: the Matern family.
"""

import math

from . import _array_core as _core
from . import _w3num as _w
from ._richresult import RichResult

__all__ = ["hyperparam_optim_gp", "hyper2", "kernel_matrix",
           "log_marginal_likelihood", "slice_sample_1d", "KERNELS",
           "ROUTES", "cheatsheet"]

KERNELS = ("squared_exponential", "matern32", "matern52")
ROUTES = ("marginal", "whitened", "surrogate")


def _dist(a, b):
    return math.sqrt(_w.csum((a[k] - b[k]) * (a[k] - b[k])
                             for k in range(len(a))))


def kernel_matrix(X, Z, log_ls, log_sf, kind="squared_exponential"):
    """Covariance between two sets of inputs.

    Parameters
    ----------
    X, Z : sequences of sequences
    log_ls : float
        log lengthscale.
    log_sf : float
        log signal standard deviation.
    kind : str
        A member of KERNELS.

    Returns
    -------
    list of list of float
    """
    if kind not in KERNELS:
        raise ValueError("kind must be one of %r" % (KERNELS,))
    ls = math.exp(log_ls)
    sf2 = math.exp(2.0 * log_sf)
    out = []
    for i in range(len(X)):
        row = []
        for j in range(len(Z)):
            r = _dist(X[i], Z[j]) / ls
            if kind == "squared_exponential":
                v = math.exp(-0.5 * r * r)
            elif kind == "matern32":
                s = math.sqrt(3.0) * r
                v = (1.0 + s) * math.exp(-s)
            else:
                s = math.sqrt(5.0) * r
                v = (1.0 + s + s * s / 3.0) * math.exp(-s)
            row.append(sf2 * v)
        out.append(row)
    return out


def _add_jitter(K, v):
    n = len(K)
    return [[K[i][j] + (v if i == j else 0.0) for j in range(n)]
            for i in range(n)]


def log_marginal_likelihood(y, X, log_ls, log_sf, log_sn, kind):
    """log N(y; 0, K + s2n I), by Cholesky.

    -0.5 y' A^-1 y - sum log diag(L) - n/2 log(2 pi), which is the form
    that never builds an inverse.
    """
    n = len(y)
    K = _add_jitter(kernel_matrix(X, X, log_ls, log_sf, kind),
                    math.exp(2.0 * log_sn) + 1e-10)
    L = _w.chol(K)
    a = _w.solve_chol(L, list(y))
    quad = _w.dot(y, a)
    logdet = 2.0 * _w.csum(math.log(L[i][i]) for i in range(n))
    return -0.5 * quad - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi)


def _log_prior(theta):
    """Standard normal on each log-hyperparameter.

    A lognormal prior, which is proper -- an improper flat prior on a log
    lengthscale gives an improper posterior whenever the data cannot
    rule out an arbitrarily long one, and a sampler will happily wander
    off into that region without ever telling you.
    """
    return _w.csum(-0.5 * v * v - 0.5 * math.log(2.0 * math.pi)
                   for v in theta)


def slice_sample_1d(logf, x0, rng, w=1.0, m=10):
    """Neal's univariate slice sampler: stepping out, then shrinkage.

    No step size to tune: `w` only sets the unit the stepping-out uses,
    and the procedure is correct for any positive value of it.
    """
    ly = logf(x0) + math.log(float(rng.uniform()))
    u = float(rng.uniform())
    lo = x0 - w * u
    hi = lo + w
    j = int(float(rng.uniform()) * m)
    k = m - 1 - j
    while j > 0 and ly < logf(lo):
        lo -= w
        j -= 1
    while k > 0 and ly < logf(hi):
        hi += w
        k -= 1
    for _ in range(200):
        x1 = lo + float(rng.uniform()) * (hi - lo)
        if ly < logf(x1):
            return x1
        if x1 < x0:
            lo = x1
        else:
            hi = x1
    return x0


def _elliptical(logl, f, L, rng):
    """Elliptical slice sampling for a latent with prior N(0, L L').

    The proposal is an exact ellipse through the current point and a
    fresh prior draw, so the prior term cancels and only the likelihood
    enters the acceptance test. There is no rejection and no step size.
    """
    n = len(f)
    nu = [float(rng.normal()) for _ in range(n)]
    v = [_w.dot(L[i][:i + 1], nu[:i + 1]) for i in range(n)]
    ly = logl(f) + math.log(float(rng.uniform()))
    a = 2.0 * math.pi * float(rng.uniform())
    amin = a - 2.0 * math.pi
    amax = a
    for _ in range(200):
        ca = math.cos(a)
        sa = math.sin(a)
        fp = [f[i] * ca + v[i] * sa for i in range(n)]
        if logl(fp) > ly:
            return fp
        if a > 0.0:
            amax = a
        else:
            amin = a
        a = amin + float(rng.uniform()) * (amax - amin)
    return f


def hyperparam_optim_gp(X, y, prior=None, kind="squared_exponential",
                        route="marginal", n_iter=200, burn=None, thin=1,
                        seed=1, Xstar=None, w=1.0, m=10, jitter=1e-8):
    """Posterior over GP hyperparameters, with predictions averaged over it.

    Parameters
    ----------
    X : sequence of sequences
        Training inputs.
    y : sequence
        Training targets.
    prior : sequence or None
        Starting values for (log lengthscale, log signal sd, log noise
        sd). The prior itself is standard normal on each; this argument
        only sets where the chain begins, and is named `prior` because
        the ledger's signature calls it that.
    kind : str
        A member of KERNELS.
    route : str
        "marginal", "whitened" or "surrogate".
    n_iter, burn, thin : int
        Chain length, burn-in and thinning.
    seed : int
        Seed for the generator shared with the R arm.
    Xstar : sequence of sequences or None
        Test inputs. Defaults to the training inputs.
    w, m : float, int
        Slice-sampling stepping-out width and step limit.
    jitter : float
        Added to the diagonal for numerical positive-definiteness.

    Returns
    -------
    RichResult
        Posterior draws of the three log-hyperparameters, their
        posterior means, the predictive mean and standard deviation at
        Xstar averaged over the draws, and the acceptance-free
        diagnostics.

    References
    ----------
    Murray and Adams (2010) NIPS 23, 1732-1740; Murray, Adams and MacKay
    (2010) AISTATS 9, 541-548; Neal (2003) Ann. Statist. 31(3), 705-767;
    Rasmussen and Williams (2006) chapters 2 and 4.
    """
    if kind not in KERNELS:
        raise ValueError("kind must be one of %r" % (KERNELS,))
    if route not in ROUTES:
        raise ValueError("route must be one of %r" % (ROUTES,))
    Xv = [[float(v) for v in row] for row in X]
    yv = [float(v) for v in y]
    n = len(yv)
    if len(Xv) != n:
        raise ValueError("X and y must have the same length")
    if n < 3:
        raise ValueError("need at least three points")
    if burn is None:
        burn = int(n_iter) // 2
    theta = ([0.0, 0.0, -1.0] if prior is None
             else [float(v) for v in prior])
    if len(theta) != 3:
        raise ValueError("prior must hold three starting log values")
    Xs = Xv if Xstar is None else [[float(v) for v in r] for r in Xstar]

    rng = _core._SplitMix64(seed)
    f = [0.0] * n
    nu = [0.0] * n

    def chol_of(th):
        return _w.chol(_add_jitter(kernel_matrix(Xv, Xv, th[0], th[1],
                                                 kind), jitter))

    def loglik(fv, sn):
        s2 = math.exp(2.0 * sn)
        return _w.csum(-0.5 * (yv[i] - fv[i]) * (yv[i] - fv[i]) / s2
                       - 0.5 * math.log(2.0 * math.pi * s2)
                       for i in range(n))

    if route != "marginal":
        L = chol_of(theta)
        nu = [float(rng.normal()) for _ in range(n)]
        f = [_w.dot(L[i][:i + 1], nu[:i + 1]) for i in range(n)]

    draws = []
    lml = []
    for it in range(int(n_iter)):
        if route == "marginal":
            for c in range(3):
                def target(v, c=c):
                    th = list(theta)
                    th[c] = v
                    return (log_marginal_likelihood(yv, Xv, th[0], th[1],
                                                    th[2], kind)
                            + _log_prior(th))
                theta[c] = slice_sample_1d(target, theta[c], rng, w, m)
            cur = log_marginal_likelihood(yv, Xv, theta[0], theta[1],
                                          theta[2], kind)
        elif route == "whitened":
            # theta moves with nu held fixed, so f = L(theta) nu follows
            # the kernel instead of being stranded by it.
            for c in range(3):
                def target(v, c=c):
                    th = list(theta)
                    th[c] = v
                    Lt = chol_of(th)
                    ft = [_w.dot(Lt[i][:i + 1], nu[:i + 1])
                          for i in range(n)]
                    return loglik(ft, th[2]) + _log_prior(th)
                theta[c] = slice_sample_1d(target, theta[c], rng, w, m)
            L = chol_of(theta)
            f = [_w.dot(L[i][:i + 1], nu[:i + 1]) for i in range(n)]
            f = _elliptical(lambda fv: loglik(fv, theta[2]), f, L, rng)
            # Recover nu from the new f so the next theta move is
            # consistent: nu = L^-1 f, by forward substitution.
            nu = [0.0] * n
            for i in range(n):
                nu[i] = (f[i] - _w.dot(L[i][:i], nu[:i])) / L[i][i]
            cur = loglik(f, theta[2]) + _log_prior(theta)
        else:
            # Surrogate data: g is a noisy view of f that theta must
            # stay consistent with. S is the likelihood's own noise,
            # which is the paper's default choice.
            s2 = math.exp(2.0 * theta[2])
            g = [f[i] + math.sqrt(s2) * float(rng.normal())
                 for i in range(n)]

            def post(th):
                Kt = _add_jitter(kernel_matrix(Xv, Xv, th[0], th[1],
                                               kind), jitter)
                s2t = math.exp(2.0 * th[2])
                A = _add_jitter(Kt, s2t)
                La = _w.chol(A)
                # m = K (K + S)^-1 g and cov = K - K (K + S)^-1 K
                sol = _w.solve_chol(La, list(g))
                mvec = [_w.dot(Kt[i], sol) for i in range(n)]
                cols = [_w.solve_chol(La, [Kt[r][j] for r in range(n)])
                        for j in range(n)]
                cov = [[Kt[i][j] - _w.dot(Kt[i], [cols[j][r]
                                                  for r in range(n)])
                        for j in range(n)] for i in range(n)]
                cov = _add_jitter(cov, jitter)
                return mvec, _w.chol(cov), La

            mvec, R, La = post(theta)
            eta = [0.0] * n
            for i in range(n):
                eta[i] = (f[i] - mvec[i] - _w.dot(R[i][:i], eta[:i])) \
                    / R[i][i]

            for c in range(3):
                def target(v, c=c):
                    th = list(theta)
                    th[c] = v
                    mv, Rt, Lat = post(th)
                    ft = [mv[i] + _w.dot(Rt[i][:i + 1], eta[:i + 1])
                          for i in range(n)]
                    sol = _w.solve_chol(Lat, list(g))
                    logdet = 2.0 * _w.csum(math.log(Lat[i][i])
                                           for i in range(n))
                    lg = (-0.5 * _w.dot(g, sol) - 0.5 * logdet
                          - 0.5 * n * math.log(2.0 * math.pi))
                    return loglik(ft, th[2]) + lg + _log_prior(th)
                theta[c] = slice_sample_1d(target, theta[c], rng, w, m)
            mvec, R, La = post(theta)
            f = [mvec[i] + _w.dot(R[i][:i + 1], eta[:i + 1])
                 for i in range(n)]
            L = chol_of(theta)
            f = _elliptical(lambda fv: loglik(fv, theta[2]), f, L, rng)
            cur = loglik(f, theta[2]) + _log_prior(theta)

        if it >= burn and (it - burn) % thin == 0:
            draws.append(list(theta))
            lml.append(cur)

    if not draws:
        raise ValueError("burn-in consumed every sweep")

    # Predictions, averaged over the hyperparameter draws. This is the
    # "integrate out" the module is for: the predictive variance picks
    # up the spread of the means across draws as well as each draw's own
    # variance, which a plug-in estimate at the maximiser cannot.
    ns = len(Xs)
    pm = [0.0] * ns
    pv = [0.0] * ns
    pm2 = [0.0] * ns
    for th in draws:
        Kxx = _add_jitter(kernel_matrix(Xv, Xv, th[0], th[1], kind),
                          math.exp(2.0 * th[2]) + jitter)
        Lx = _w.chol(Kxx)
        alpha = _w.solve_chol(Lx, list(yv))
        Ksx = kernel_matrix(Xs, Xv, th[0], th[1], kind)
        Kss = kernel_matrix(Xs, Xs, th[0], th[1], kind)
        for j in range(ns):
            mj = _w.dot(Ksx[j], alpha)
            vj = _w.solve_chol(Lx, list(Ksx[j]))
            sj = Kss[j][j] - _w.dot(Ksx[j], vj)
            if sj < 0.0:
                sj = 0.0
            pm[j] += mj
            pm2[j] += mj * mj
            pv[j] += sj
    M = len(draws)
    mean = [v / M for v in pm]
    var = [pv[j] / M + pm2[j] / M - mean[j] * mean[j] for j in range(ns)]
    sd = [math.sqrt(v if v > 0.0 else 0.0) for v in var]

    means = [_w.csum(d[c] for d in draws) / M for c in range(3)]
    sds = []
    for c in range(3):
        if M > 1:
            sds.append(math.sqrt(_w.csum((d[c] - means[c])
                                         * (d[c] - means[c])
                                         for d in draws) / (M - 1)))
        else:
            sds.append(0.0)

    return RichResult(payload={
        "draws": draws,
        "log_lengthscale": means[0],
        "log_signal_sd": means[1],
        "log_noise_sd": means[2],
        "lengthscale": math.exp(means[0]),
        "signal_sd": math.exp(means[1]),
        "noise_sd": math.exp(means[2]),
        "posterior_sd": sds,
        "log_target": lml,
        "mean_log_target": _w.csum(lml) / M,
        "predict_mean": mean,
        "predict_sd": sd,
        "n": n,
        "n_test": ns,
        "kept": M,
        "kind": kind,
        "route": route,
        "seed": int(seed),
        "estimate": means[0],
        "method": "GP hyperparameter MCMC with the latent integrated out",
    })


hyper2 = hyperparam_optim_gp


def cheatsheet():
    return ("hyper2: GP hyperparameter MCMC. kernels "
            + ", ".join(KERNELS) + "; routes " + ", ".join(ROUTES))
