"""morie ts core: statsmodels time-series subset.

ARIMA / SARIMAX via the Harvey state-space representation with exact
Kalman-filter Gaussian likelihood (stationary initialization from the
discrete Lyapunov equation); UnobservedComponents (local level +
seasonal dummies) with Kalman smoothing; MarkovRegression (Hamilton
filter + Kim smoother, EM); Johansen cointegration (eigenproblem +
Osterwald-Lenum critical values) and VECM. Equivalence-tested against
statsmodels in tests/fn/test_ts_core.py.
"""

from __future__ import annotations

import builtins as _bi
import math as _math

from . import _array_core as _ac
from ._sci_core import minimize as _minimize


def _y1(y):
    return [float(v) for v in _ac.asarray(y)._flat()]


# ------------------------------------------------------------ kalman

def _mat(v, r, c):
    return [[float(v)] * c for _ in range(r)] if not isinstance(
        v, list) else v


def _mm(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[_math.fsum(A[i][k] * B[k][j] for k in range(m))
             for j in range(p)] for i in range(n)]


def _mt(A):
    return [[A[j][i] for j in range(len(A))]
            for i in range(len(A[0]))]


def _madd(A, B, s=1.0):
    return [[A[i][j] + s * B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def _solve_discrete_lyapunov(T, Q):
    """P = T P T' + Q via the vec/Kronecker linear system."""
    m = len(T)
    n2 = m * m
    big = [[(1.0 if a == b else 0.0) for b in range(n2)]
           for a in range(n2)]
    for i in range(m):
        for j in range(m):
            r = i * m + j
            for k in range(m):
                for l_ in range(m):
                    big[r][k * m + l_] -= T[i][k] * T[j][l_]
    rhs = [Q[i][j] for i in range(m) for j in range(m)]
    x = _ac.linalg.solve(_ac.marr(big), _ac.marr(rhs))
    xv = list(x._flat())
    return [[xv[i * m + j] for j in range(m)] for i in range(m)]


def _kalman_loglik(y, T, Z, RQR, H, P0=None, a0=None,
                   return_states=False, burn=0):
    """Exact Gaussian loglik for a_t+1 = T a_t + eta,
    y_t = Z a_t + eps; RQR = state innovation covariance,
    H = observation variance (scalar)."""
    m = len(T)
    a = [0.0] * m if a0 is None else list(a0)
    if P0 is None:
        try:
            P = _solve_discrete_lyapunov(T, RQR)
        except Exception:
            P = [[1e6 if i == j else 0.0 for j in range(m)]
                 for i in range(m)]
    else:
        P = [row[:] for row in P0]
    ll = 0.0
    n = len(y)
    filt_a = []
    filt_P = []
    pred_a = []
    pred_P = []
    for t in range(n):
        pred_a.append(list(a))
        pred_P.append([row[:] for row in P])
        # innovation
        v = y[t] - _math.fsum(Z[j] * a[j] for j in range(m))
        PZ = [_math.fsum(P[i][j] * Z[j] for j in range(m))
              for i in range(m)]
        F = _math.fsum(Z[i] * PZ[i] for i in range(m)) + H
        if F <= 1e-12:
            F = 1e-12
        if t >= burn:
            ll += -0.5 * (_math.log(2.0 * _math.pi) + _math.log(F)
                          + v * v / F)
        K = [PZ[i] / F for i in range(m)]
        a = [a[i] + K[i] * v for i in range(m)]
        P = [[P[i][j] - K[i] * PZ[j] for j in range(m)]
             for i in range(m)]
        filt_a.append(list(a))
        filt_P.append([row[:] for row in P])
        # time update
        a = [_math.fsum(T[i][j] * a[j] for j in range(m))
             for i in range(m)]
        P = _madd(_mm(_mm(T, P), _mt(T)), RQR)
    if not return_states:
        return ll
    return ll, filt_a, filt_P, pred_a, pred_P


def _kalman_smooth(y, T, Z, RQR, H):
    ll, fa, fP, pa, pP = _kalman_loglik(y, T, Z, RQR, H,
                                        return_states=True)
    n = len(y)
    m = len(T)
    sm_a = [None] * n
    sm_a[n - 1] = list(fa[n - 1])
    Pnext = None
    a_sm = list(fa[n - 1])
    P_sm = [row[:] for row in fP[n - 1]]
    for t in range(n - 2, -1, -1):
        # J = fP[t] T' pred_P[t+1]^-1
        Ppred = pP[t + 1]
        Pinv = _ac.linalg.inv(_ac.marr(
            _madd(Ppred, [[1e-12 if i == j else 0.0
                           for j in range(m)] for i in range(m)])
        )).tolist()
        J = _mm(_mm(fP[t], _mt(T)), Pinv)
        diff = [a_sm[i] - pa[t + 1][i] for i in range(m)]
        a_sm = [fa[t][i] + _math.fsum(J[i][j] * diff[j]
                                      for j in range(m))
                for i in range(m)]
        sm_a[t] = list(a_sm)
    del Pnext, P_sm
    return ll, sm_a


# ------------------------------------------------------------ ARMA ss

def _arma_ss(phi, theta, sigma2):
    p, q = len(phi), len(theta)
    m = _bi.max(p, q + 1)
    T = [[0.0] * m for _ in range(m)]
    for i in range(p):
        T[i][0] = phi[i]
    for i in range(m - 1):
        T[i][i + 1] = 1.0
    # Harvey: state transition uses phi in first column,
    # superdiagonal identity
    R = [1.0] + [theta[i] if i < q else 0.0 for i in range(m - 1)]
    RQR = [[sigma2 * R[i] * R[j] for j in range(m)]
           for i in range(m)]
    Z = [1.0] + [0.0] * (m - 1)
    return T, Z, RQR


def _diff(y, d):
    out = list(y)
    for _ in range(d):
        out = [out[i + 1] - out[i] for i in range(len(out) - 1)]
    return out


def _sdiff(y, D, s):
    out = list(y)
    for _ in range(D):
        out = [out[i + s] - out[i] for i in range(len(out) - s)]
    return out


def _polymul_coefs(a, b):
    """(1 - a1 B - ...)(1 - b1 B^s ...) style products handled by the
    caller; here plain coefficient convolution of [1, c1, c2, ...]."""
    out = [0.0] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            out[i + j] += av * bv
    return out


class _ARIMAResults:
    def __init__(self, model, params, llf, k):
        self.model = model
        self.params = _ac.marr(params)
        self.llf = llf
        self.aic = -2.0 * llf + 2.0 * k
        self.bic = -2.0 * llf + k * _math.log(len(model.y_orig))

    def forecast(self, steps=1):
        return self.model._forecast(list(self.params._flat()), steps)

    @property
    def fittedvalues(self):
        return self.model._fitted(list(self.params._flat()))


class ARIMA:
    """ARIMA(p, d, q) with exact state-space likelihood."""

    def __init__(self, endog, order=(1, 0, 0), **kw):
        del kw
        self.y_orig = _y1(endog)
        self.p, self.d, self.q = order
        self.w = _diff(self.y_orig, self.d)
        self.seasonal = None

    def _unpack(self, params):
        p, q = self.p, self.q
        phi = list(params[:p])
        theta = list(params[p:p + q])
        sigma2 = _math.exp(params[p + q])
        return phi, theta, sigma2

    def _negll(self, params):
        phi, theta, sigma2 = self._unpack(params)
        # soft stationarity guard
        if _math.fsum(abs(v) for v in phi) > 10 or \
                _math.fsum(abs(v) for v in theta) > 10:
            return 1e10
        try:
            T, Z, RQR = _arma_ss(phi, theta, sigma2)
            return -_kalman_loglik(self.w, T, Z, RQR, 0.0)
        except Exception:
            return 1e10

    def fit(self, **kw):
        del kw
        p, q = self.p, self.q
        var_w = (_math.fsum(v * v for v in self.w) / len(self.w)
                 - (_math.fsum(self.w) / len(self.w)) ** 2) or 1.0
        x0 = [0.1] * p + [0.1] * q + [_math.log(var_w)]
        res = _minimize(self._negll, x0, method="Nelder-Mead",
                        options={"maxiter": 4000, "xatol": 1e-8,
                                 "fatol": 1e-8})
        params = list(res.x._flat())
        llf = -float(res.fun)
        return _ARIMAResults(self, params, llf, p + q + 1)

    def _filter_resid(self, params):
        phi, theta, sigma2 = self._unpack(params)
        T, Z, RQR = _arma_ss(phi, theta, sigma2)
        _ll, fa, _fP, pa, _pP = _kalman_loglik(
            self.w, T, Z, RQR, 0.0, return_states=True)
        return phi, theta, fa, pa

    def _forecast(self, params, steps):
        phi, theta, fa, _pa = self._filter_resid(params)
        m = len(fa[0])
        T, Z, _ = _arma_ss(phi, theta, 1.0)
        a = list(fa[-1])
        wfc = []
        for _ in range(steps):
            a = [_math.fsum(T[i][j] * a[j] for j in range(m))
                 for i in range(m)]
            wfc.append(a[0])
        # integrate the d-th differences back
        out = []
        hist = list(self.y_orig)
        if self.d == 0:
            out = wfc
        elif self.d == 1:
            level = hist[-1]
            for v in wfc:
                level += v
                out.append(level)
        else:
            # general d: cumulative reconstruction
            tails = [hist]
            for _k in range(self.d):
                tails.append(_diff(tails[-1], 1))
            lasts = [t[-1] for t in tails[:-1]]
            for v in wfc:
                new = v
                for k in range(self.d - 1, -1, -1):
                    new = lasts[k] + new
                    lasts[k] = new
                out.append(new)
        return _ac.marr(out)

    def _fitted(self, params):
        phi, theta, _fa, pa = self._filter_resid(params)
        pred_w = [row[0] for row in pa]
        if self.d == 0:
            return _ac.marr(pred_w)
        base = self.y_orig
        return _ac.marr([base[i] + pred_w[i]
                         for i in range(len(pred_w))])


class SARIMAX(ARIMA):
    """Multiplicative seasonal ARIMA: coefficients expanded into a
    single high-order ARMA, exact Kalman likelihood."""

    def __init__(self, endog, order=(1, 0, 0),
                 seasonal_order=(0, 0, 0, 0), **kw):
        del kw
        self.y_orig = _y1(endog)
        self.p, self.d, self.q = order
        self.P, self.D, self.Q, self.s = seasonal_order
        w = _diff(self.y_orig, self.d)
        if self.D and self.s:
            w = _sdiff(w, self.D, self.s)
        self.w = w

    def _unpack(self, params):
        p, q, P, Q, s = self.p, self.q, self.P, self.Q, self.s
        pos = 0
        phi = list(params[pos:pos + p]); pos += p
        theta = list(params[pos:pos + q]); pos += q
        Phi = list(params[pos:pos + P]); pos += P
        Theta = list(params[pos:pos + Q]); pos += Q
        sigma2 = _math.exp(params[pos])
        # expand multiplicative polynomials
        arp = [1.0] + [-v for v in phi]
        sar = [1.0] + [0.0] * (P * s)
        for k in range(P):
            sar[(k + 1) * s] = -Phi[k]
        full_ar = _polymul_coefs(arp, sar)
        map_ = [1.0] + list(theta)
        sma = [1.0] + [0.0] * (Q * s)
        for k in range(Q):
            sma[(k + 1) * s] = Theta[k]
        full_ma = _polymul_coefs(map_, sma)
        phi_full = [-v for v in full_ar[1:]]
        theta_full = list(full_ma[1:])
        return phi_full, theta_full, sigma2

    def _npar(self):
        return self.p + self.q + self.P + self.Q + 1

    def _negll(self, params):
        phi, theta, sigma2 = self._unpack(params)
        try:
            T, Z, RQR = _arma_ss(phi, theta, sigma2)
            # non-stationary allowed (enforce_stationarity=False):
            # use large diffuse-ish P0 if lyapunov fails
            return -_kalman_loglik(self.w, T, Z, RQR, 0.0)
        except Exception:
            return 1e10

    def fit(self, **kw):
        del kw
        var_w = (_math.fsum(v * v for v in self.w) / len(self.w)
                 or 1.0)
        x0 = [0.1] * (self.p + self.q + self.P + self.Q) \
            + [_math.log(var_w)]
        res = _minimize(self._negll, x0, method="Nelder-Mead",
                        options={"maxiter": 6000})
        params = list(res.x._flat())
        return _ARIMAResults(self, params, -float(res.fun),
                             self._npar())

    def _forecast(self, params, steps):
        phi, theta, fa, _pa = self._filter_resid(params)
        m = len(fa[0])
        T, Z, _ = _arma_ss(phi, theta, 1.0)
        a = list(fa[-1])
        wfc = []
        for _ in range(steps):
            a = [_math.fsum(T[i][j] * a[j] for j in range(m))
                 for i in range(m)]
            wfc.append(a[0])
        # undo seasonal then regular differencing
        w_hist = _diff(self.y_orig, self.d)
        if self.D and self.s:
            # x_t = w_t + x_{t-s} on the d-differenced scale
            xd = list(w_hist)
            out_d = []
            for k, v in enumerate(wfc):
                nxt = v + xd[len(xd) - self.s]
                xd.append(nxt)
                out_d.append(nxt)
        else:
            out_d = wfc
        if self.d == 0:
            return _ac.marr(out_d)
        level = self.y_orig[-1]
        out = []
        for v in out_d:
            level += v
            out.append(level)
        return _ac.marr(out)


# ------------------------------------------------------------ UC

class _UCResults:
    def __init__(self, params, llf, level_sm, seasonal_sm):
        self.params = _ac.marr(params)
        self.llf = llf
        self.level = {"smoothed": _ac.marr(level_sm)}
        self.seasonal = ({"smoothed": _ac.marr(seasonal_sm)}
                         if seasonal_sm is not None else None)


class UnobservedComponents:
    """Local level (+ optional dummy seasonal) structural model."""

    def __init__(self, endog, level="local level", seasonal=None,
                 freq_seasonal=None, **kw):
        del freq_seasonal, kw
        self.y = _y1(endog)
        self.level = level
        self.period = int(seasonal) if seasonal else 0

    def _system(self, sig_eps, sig_level, sig_seas):
        s = self.period
        if s and s > 1:
            m = 1 + (s - 1)
            T = [[0.0] * m for _ in range(m)]
            T[0][0] = 1.0
            for j in range(1, s):
                T[1][j] = -1.0
            for i in range(2, m):
                T[i][i - 1] = 1.0
            Z = [1.0, 1.0] + [0.0] * (m - 2)
            RQR = [[0.0] * m for _ in range(m)]
            RQR[0][0] = sig_level
            RQR[1][1] = sig_seas
        else:
            T = [[1.0]]
            Z = [1.0]
            RQR = [[sig_level]]
        return T, Z, RQR

    def _negll(self, params):
        vals = [_math.exp(v) for v in params]
        sig_eps = vals[0]
        sig_level = vals[1]
        sig_seas = vals[2] if len(vals) > 2 else 0.0
        try:
            T, Z, RQR = self._system(sig_eps, sig_level, sig_seas)
            m = len(T)
            P0 = [[1e7 if i == j else 0.0 for j in range(m)]
                  for i in range(m)]
            # approximate diffuse init: burn the first m innovation
            # terms like the exact-diffuse loglik does
            return -_kalman_loglik(self.y, T, Z, RQR, sig_eps,
                                   P0=P0, burn=m)
        except Exception:
            return 1e10

    def fit(self, **kw):
        del kw
        var_y = (_math.fsum(v * v for v in self.y) / len(self.y)
                 - (_math.fsum(self.y) / len(self.y)) ** 2) or 1.0
        k = 3 if self.period and self.period > 1 else 2
        x0 = [_math.log(var_y / 2.0)] * k
        res = _minimize(self._negll, x0, method="Nelder-Mead",
                        options={"maxiter": 3000})
        pv = [_math.exp(v) for v in res.x._flat()]
        sig_eps, sig_level = pv[0], pv[1]
        sig_seas = pv[2] if len(pv) > 2 else 0.0
        T, Z, RQR = self._system(sig_eps, sig_level, sig_seas)
        m = len(T)
        # smoothing with diffuse-ish init
        ll, sm_a = _kalman_smooth(self.y, T, Z, RQR, sig_eps)
        level_sm = [a[0] for a in sm_a]
        seasonal_sm = [a[1] for a in sm_a] if m > 1 else None
        # statsmodels param order: sigma2.irregular, sigma2.level,
        # (sigma2.seasonal)
        params = [sig_eps, sig_level] + (
            [sig_seas] if m > 1 else [])
        return _UCResults(params, -float(self._negll(
            [_math.log(v + 1e-300) for v in params])), level_sm,
            seasonal_sm)


# ------------------------------------------------------------ markov

class _MarkovParams(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key)


class _MarkovResults:
    def __init__(self, params, P, smoothed, llf, k):
        self.params = _MarkovParams(params)
        self.regime_transition = _ac.marr(P)
        self.smoothed_marginal_probabilities = _ac.marr(smoothed)
        self.llf = llf
        self.aic = -2.0 * llf + 2.0 * k


class MarkovRegression:
    """Gaussian mixture-of-regimes mean/variance switching model,
    estimated by EM on the Hamilton filter / Kim smoother."""

    def __init__(self, endog, k_regimes=2, switching_variance=True,
                 **kw):
        del kw
        self.y = _y1(endog)
        self.k = k_regimes
        self.sv = switching_variance

    def fit(self, maxiter=200, tol=1e-8, **kw):
        del kw
        y = self.y
        n = len(y)
        k = self.k
        # init: split by quantiles
        ys = sorted(y)
        mu = [ys[int((i + 0.5) * n / k)] for i in range(k)]
        var_all = (_math.fsum(v * v for v in y) / n
                   - (_math.fsum(y) / n) ** 2) or 1.0
        sig2 = [var_all] * k
        P = [[0.9 if i == j else 0.1 / (k - 1) for j in range(k)]
             for i in range(k)]
        llf_old = -_math.inf
        for _it in range(maxiter):
            # Hamilton filter
            dens = [[_math.exp(-0.5 * (y[t] - mu[j]) ** 2
                               / sig2[j])
                     / _math.sqrt(2.0 * _math.pi * sig2[j])
                     for j in range(k)] for t in range(n)]
            # stationary init
            pi0 = [1.0 / k] * k
            filt = []
            pred = pi0
            llf = 0.0
            for t in range(n):
                joint = [pred[j] * dens[t][j] for j in range(k)]
                s = _math.fsum(joint) + 1e-300
                llf += _math.log(s)
                f = [v / s for v in joint]
                filt.append(f)
                pred = [_math.fsum(f[i] * P[i][j]
                                   for i in range(k))
                        for j in range(k)]
            # Kim smoother
            smooth = [None] * n
            smooth[n - 1] = filt[n - 1]
            xi = [[[0.0] * k for _ in range(k)]
                  for _ in range(n - 1)]
            for t in range(n - 2, -1, -1):
                prednext = [_math.fsum(filt[t][i] * P[i][j]
                                       for i in range(k))
                            for j in range(k)]
                sm = [0.0] * k
                for i in range(k):
                    for j in range(k):
                        w = filt[t][i] * P[i][j] \
                            * smooth[t + 1][j] \
                            / (prednext[j] + 1e-300)
                        xi[t][i][j] = w
                        sm[i] += w
                smooth[t] = sm
            # M step
            for j in range(k):
                wj = _math.fsum(smooth[t][j] for t in range(n))
                mu[j] = _math.fsum(smooth[t][j] * y[t]
                                   for t in range(n)) / (wj + 1e-300)
                if self.sv:
                    sig2[j] = _math.fsum(
                        smooth[t][j] * (y[t] - mu[j]) ** 2
                        for t in range(n)) / (wj + 1e-300)
                    sig2[j] = _bi.max(sig2[j], 1e-8)
            if not self.sv:
                pooled = _math.fsum(
                    smooth[t][j] * (y[t] - mu[j]) ** 2
                    for t in range(n) for j in range(k)) / n
                sig2 = [pooled] * k
            for i in range(k):
                denom = _math.fsum(xi[t][i][j] for t in range(n - 1)
                                   for j in range(k)) + 1e-300
                for j in range(k):
                    P[i][j] = _math.fsum(xi[t][i][j]
                                         for t in range(n - 1)) \
                        / denom
            if abs(llf - llf_old) < tol * (abs(llf) + 1.0):
                break
            llf_old = llf
        # order regimes by mean for stable labels
        order = sorted(range(k), key=lambda j: mu[j])
        mu = [mu[j] for j in order]
        sig2 = [sig2[j] for j in order]
        P = [[P[order[i]][order[j]] for j in range(k)]
             for i in range(k)]
        smooth = [[smooth[t][order[j]] for j in range(k)]
                  for t in range(n)]
        params = {}
        for j in range(k):
            params["const[%d]" % j] = mu[j]
            params["sigma2[%d]" % j] = sig2[j]
        npar = k * 2 + k * (k - 1)
        return _MarkovResults(params, P, smooth, llf, npar)


# ------------------------------------------------------------ johansen

# Trace critical values for det_order=0 (constant), rows are
# (n - r) = 1..10, columns 90 / 95 / 99 percent — values read directly
# from statsmodels coint_johansen output (its embedded c_sjt table).
_OL_TRACE_CV = [
    [2.7055, 3.8415, 6.6349],
    [13.4294, 15.4943, 19.9349],
    [27.0669, 29.7961, 35.4628],
    [44.4929, 47.8545, 54.6815],
    [65.8202, 69.8189, 77.8202],
    [91.109, 95.7542, 104.9637],
    [120.3673, 125.6185, 135.9825],
    [153.6341, 159.529, 171.0905],
    [190.8714, 197.3772, 210.0366],
    [232.103, 239.2468, 253.2526],
]


class _JohansenResult:
    pass


def coint_johansen(Y, det_order=0, k_ar_diff=1):
    Ym = _ac.atleast_2d(Y)
    data = [list(map(float, r)) for r in Ym.data]
    n_obs = len(data)
    k = len(data[0])
    p = k_ar_diff
    dY = [[data[t + 1][j] - data[t][j] for j in range(k)]
          for t in range(n_obs - 1)]
    T = len(dY) - p
    # regressors: lagged differences (+ constant for det_order == 0)
    Zrows = []
    for t in range(p, len(dY)):
        row = []
        for lag in range(1, p + 1):
            row += dY[t - lag]
        if det_order == 0:
            row.append(1.0)
        Zrows.append(row)
    R0rows = [dY[t] for t in range(p, len(dY))]
    R1rows = [data[t] for t in range(p, len(dY))]   # y_{t-1} level

    def residualize(Xrows):
        if not Zrows or not Zrows[0]:
            return [row[:] for row in Xrows]
        kz = len(Zrows[0])
        ZtZ = [[_math.fsum(Zrows[t][a] * Zrows[t][b]
                           for t in range(T)) for b in range(kz)]
               for a in range(kz)]
        out = []
        kx = len(Xrows[0])
        B = []
        for j in range(kx):
            Ztx = [_math.fsum(Zrows[t][a] * Xrows[t][j]
                              for t in range(T))
                   for a in range(kz)]
            B.append(list(_ac.linalg.solve(
                _ac.marr(ZtZ), _ac.marr(Ztx))._flat()))
        for t in range(T):
            out.append([Xrows[t][j] - _math.fsum(
                Zrows[t][a] * B[j][a] for a in range(kz))
                for j in range(kx)])
        return out
    R0 = residualize(R0rows)
    R1 = residualize(R1rows)

    def S(A, B):
        ka, kb = len(A[0]), len(B[0])
        return [[_math.fsum(A[t][i] * B[t][j] for t in range(T)) / T
                 for j in range(kb)] for i in range(ka)]
    S00 = S(R0, R0)
    S11 = S(R1, R1)
    S01 = S(R0, R1)
    S10 = _mt(S01)
    S00inv = _ac.linalg.inv(_ac.marr(S00)).tolist()
    S11inv = _ac.linalg.inv(_ac.marr(S11)).tolist()
    M = _mm(_mm(S11inv, S10), _mm(S00inv, S01))
    from ._sci_core import eigvals as _eigvals
    ev = _eigvals(_ac.marr(M)).tolist()
    eig = sorted((_bi.max(0.0, _bi.min(0.999999, v.real))
                  for v in ev), reverse=True)[:k]
    lr1 = []
    for r in range(k):
        lr1.append(-T * _math.fsum(_math.log(1.0 - eig[j])
                                   for j in range(r, k)))
    cvt = [_OL_TRACE_CV[k - r - 1] for r in range(k)]
    res = _JohansenResult()
    res.eig = _ac.marr(eig)
    res.lr1 = _ac.marr(lr1)
    res.cvt = _ac.marr(cvt)
    # eigenvectors for beta: solve (M - lambda I) v = 0 via inverse
    # iteration on each eigenvalue
    vecs = []
    for lam in eig:
        A = [[M[i][j] - (lam - 1e-10 if i == j else 0.0)
              for j in range(k)] for i in range(k)]
        v = [1.0] * k
        for _ in range(50):
            try:
                v = list(_ac.linalg.solve(_ac.marr(A),
                                          _ac.marr(v))._flat())
            except Exception:
                break
            nrm = _math.sqrt(_math.fsum(u * u for u in v)) or 1.0
            v = [u / nrm for u in v]
        vecs.append(v)
    res.evec = _ac.marr([[vecs[j][i] for j in range(k)]
                         for i in range(k)])
    res._internals = (R0, R1, T, k)
    return res


class _VECMResults:
    pass


class VECM:
    def __init__(self, endog, k_ar_diff=1, coint_rank=1,
                 deterministic="ci", **kw):
        del deterministic, kw
        self.Y = _ac.atleast_2d(endog)
        self.k_ar_diff = k_ar_diff
        self.rank = coint_rank

    def fit(self, **kw):
        del kw
        j = coint_johansen(self.Y, det_order=0,
                           k_ar_diff=self.k_ar_diff)
        R0, R1, T, k = j._internals
        r = self.rank
        evec = j.evec.tolist()
        beta = [[evec[i][c] for c in range(r)] for i in range(k)]
        # alpha via OLS of R0 on (R1 beta)
        Wrows = [[_math.fsum(R1[t][i] * beta[i][c]
                             for i in range(k)) for c in range(r)]
                 for t in range(T)]
        WtW = [[_math.fsum(Wrows[t][a] * Wrows[t][b]
                           for t in range(T)) for b in range(r)]
               for a in range(r)]
        alpha = []
        for jcol in range(k):
            Wty = [_math.fsum(Wrows[t][a] * R0[t][jcol]
                              for t in range(T)) for a in range(r)]
            arow = list(_ac.linalg.solve(_ac.marr(WtW),
                                         _ac.marr(Wty))._flat())
            alpha.append(arow)
        res = _VECMResults()
        res.alpha = _ac.marr(alpha)
        res.beta = _ac.marr(beta)
        # short-run Gamma via OLS of dY on lagged dY given ECT
        data = [list(map(float, row)) for row in self.Y.data]
        n_obs = len(data)
        p = self.k_ar_diff
        dY = [[data[t + 1][c] - data[t][c] for c in range(k)]
              for t in range(n_obs - 1)]
        rows = []
        targets = []
        for t in range(p, len(dY)):
            ect = [_math.fsum(data[t][i] * beta[i][c]
                              for i in range(k)) for c in range(r)]
            row = list(ect)
            for lag in range(1, p + 1):
                row += dY[t - lag]
            row.append(1.0)
            rows.append(row)
            targets.append(dY[t])
        kz = len(rows[0])
        ZtZ = [[_math.fsum(rows[t][a] * rows[t][b]
                           for t in range(len(rows)))
                for b in range(kz)] for a in range(kz)]
        gam = []
        for c in range(k):
            Zty = [_math.fsum(rows[t][a] * targets[t][c]
                              for t in range(len(rows)))
                   for a in range(kz)]
            coef = list(_ac.linalg.solve(_ac.marr(ZtZ),
                                         _ac.marr(Zty))._flat())
            gam.append(coef[r:r + p * k])
        res.gamma = _ac.marr(gam)
        return res


class _TsaStatespace:
    SARIMAX = SARIMAX


class tsa:  # namespace mirror for sm.tsa
    ARIMA = ARIMA
    statespace = _TsaStatespace()
