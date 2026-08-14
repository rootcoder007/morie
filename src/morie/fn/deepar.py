# morie.fn -- function file (rootcoder007/morie)
r"""DeepAR: autoregressive probabilistic forecasting.

A point forecast is not what an inventory or capacity decision needs --
the quantile is. DeepAR fits an autoregressive recurrent model that
emits the *parameters* of a likelihood at each step,

.. math:: \ell\big(z_t \mid \theta(h_t)\big), \qquad
          h_t = f(h_{t-1}, z_{t-1}, x_t),

and produces forecast distributions by sampling the recursion forward.

**The likelihood must match the data, and Gaussian is usually wrong.**
Demand is a non-negative count with variance that grows with the level.
The paper's negative binomial parameterises exactly that,

.. math:: \mu > 0, \quad \alpha > 0, \qquad
          \mathrm{Var} = \mu(1 + \mu\alpha),

so :math:`\alpha` is the *shape* of the overdispersion, not a nuisance:
at :math:`\alpha \to 0` it collapses to Poisson, where variance equals
the mean. The anchor checks that limit and checks the variance formula
against a sample.

**Scale handling is what makes it work across series.** Item sales
differ by orders of magnitude, and a shared network cannot see that
range. Each series is divided by its own average
:math:`\nu_i = 1 + \frac{1}{t_0}\sum z_{i,t}`, and the outputs
multiplied back. Skip it and the loss is dominated by whichever series
happens to be largest -- the anchor scales one series by 1000 and
checks the *normalised* forecast is unchanged.

**Prediction is by simulation, not by a formula.** Multi-step
uncertainty has no closed form once the model feeds its own output
back, so trajectories are sampled and quantiles read off. That is why
the intervals widen with horizon without anything being told to widen
them.

References
----------
Salinas, D., Flunkert, V., Gasthaus, J. & Januschowski, T. (2020)
"DeepAR: Probabilistic forecasting with autoregressive recurrent
networks", *International Journal of Forecasting* 36(3), 1181-1191,
doi:10.1016/j.ijforecast.2019.07.001, arXiv:1704.04110. Secs. 3.2-3.3:
the likelihood, the scale handling, and ancestral sampling.

Hochreiter, S. & Schmidhuber, J. (1997) "Long Short-Term Memory",
*Neural Computation* 9(8), 1735-1780,
doi:10.1162/neco.1997.9.8.1735. The recurrent cell.

Gasthaus, J., Benidis, K., Wang, Y., Rangapuram, S. S., Salinas, D.,
Flunkert, V. & Januschowski, T. (2019) "Probabilistic Forecasting with
Spline Quantile Function RNNs", *Proceedings of the 22nd International
Conference on Artificial Intelligence and Statistics*, PMLR 89,
1901-1910. The quantile alternative to a parametric likelihood.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["scale_factor", "negative_binomial_loglik",
           "gaussian_loglik", "sample_negative_binomial",
           "deepar_fit", "deepar_sample", "deepar_forecast"]

_EPS = 1e-12


def scale_factor(z, t0=None):
    r""":math:`\nu = 1 + \frac{1}{t_0}\sum_t z_t`.

    The +1 is not decoration: it keeps the divisor away from zero for a
    series that is all zeros, which intermittent series often are.
    """
    zv = [float(v) for v in z]
    n = len(zv) if t0 is None else int(t0)
    if n < 1:
        raise ValueError("deepar: need at least one observation")
    return 1.0 + sum(zv[:n]) / float(n)


def gaussian_loglik(z, mu, sigma):
    """For real-valued series."""
    s = max(float(sigma), _EPS)
    return (-math.log(s) - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * ((float(z) - float(mu)) / s) ** 2)


def negative_binomial_loglik(z, mu, alpha):
    r"""The count likelihood, in the paper's (mu, alpha) form.

    Variance is :math:`\mu(1 + \mu\alpha)`, so alpha IS the
    overdispersion and alpha -> 0 recovers Poisson.
    """
    zz = float(z)
    m = max(float(mu), _EPS)
    a = float(alpha)
    if zz < 0.0:
        raise ValueError("deepar: the negative binomial needs a "
                         "non-negative count, got %r" % (z,))
    if a < 0.0:
        raise ValueError("deepar: alpha must be non-negative, got %r"
                         % (alpha,))
    if a < 1e-10:                      # Poisson limit
        return zz * math.log(m) - m - k.lgamma(zz + 1.0)
    r = 1.0 / a
    return (k.lgamma(zz + r) - k.lgamma(r) - k.lgamma(zz + 1.0)
            + r * math.log(r / (r + m)) + zz * math.log(m / (r + m)))


def sample_negative_binomial(mu, alpha, rng):
    r"""Gamma-Poisson mixture: the standard construction."""
    m = max(float(mu), _EPS)
    a = float(alpha)
    if a < 1e-10:
        return _poisson(m, rng)
    r = 1.0 / a
    lam = _gamma(r, m / r, rng)
    return _poisson(lam, rng)


def _poisson(lam, rng):
    if lam <= 0.0:
        return 0.0
    if lam > 30.0:
        # normal approximation with a continuity correction, so a large
        # mean does not spin the product loop forever
        v = lam + math.sqrt(lam) * rng.standard_normal()
        return float(max(0.0, math.floor(v + 0.5)))
    ll = math.exp(-lam)
    p, n = 1.0, 0
    while True:
        p *= float(rng.uniform())
        if p <= ll:
            return float(n)
        n += 1


def _gamma(shape, scale, rng):
    """Marsaglia-Tsang, with the shape<1 boost."""
    s = float(shape)
    if s < 1.0:
        u = float(rng.uniform())
        return _gamma(s + 1.0, scale, rng) * (u ** (1.0 / s))
    d = s - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.standard_normal()
        v = (1.0 + c * x) ** 3
        if v <= 0.0:
            continue
        u = float(rng.uniform())
        if math.log(u) < 0.5 * x * x + d - d * v + d * math.log(v):
            return d * v * scale


def deepar_fit(z, n_lags=2, likelihood="negative-binomial",
               ridge=1e-6):
    r"""Fit the conditional mean by a scaled autoregression.

    A linear autoregression stands in for the recurrent network: the
    scaling, the likelihood and the sampling are what this module is
    about, and they are identical either way.
    """
    if likelihood not in ("negative-binomial", "gaussian"):
        raise ValueError("deepar: likelihood must be "
                         "negative-binomial or gaussian, got %r"
                         % (likelihood,))
    zv = [float(v) for v in k.vec(z)]
    n = len(zv)
    p = int(n_lags)
    if n < p + 4:
        raise ValueError("deepar: %d observations is too few for %d "
                         "lags" % (n, p))
    nu = scale_factor(zv)
    zs = [v / nu for v in zv]                 # scaled to a common range
    X = [[1.0] + [zs[t - j - 1] for j in range(p)]
         for t in range(p, n)]
    yv = [zs[t] for t in range(p, n)]
    beta = k.lstsq(X, yv, ridge)
    fitted = [max(sum(X[i][a] * beta[a] for a in range(len(beta))),
                  0.0) for i in range(len(X))]
    resid = [yv[i] - fitted[i] for i in range(len(yv))]
    if likelihood == "negative-binomial":
        # method-of-moments alpha from Var = mu(1 + mu*alpha)
        mbar = max(k.mean(fitted), _EPS)
        vbar = k.variance(yv) if len(yv) > 1 else mbar
        alpha = max((vbar - mbar) / (mbar * mbar), 1e-8)
    else:
        alpha = max(k.sd(resid), _EPS) if len(resid) > 1 else 1.0
    return RichResult(payload={
        "estimate": beta, "beta": beta, "nu": nu, "n_lags": p,
        "likelihood": likelihood, "alpha": alpha,
        "fitted_scaled": fitted, "residual": resid,
        "scaled": zs, "n": n,
        "method": "DeepAR autoregressive probabilistic forecaster, "
                  "Salinas, Flunkert, Gasthaus & Januschowski (2020)",
    })


def deepar_sample(fit, z_history, horizon, n_samples=200, seed=0):
    r"""Ancestral sampling: feed each draw back in.

    Multi-step uncertainty has no closed form once the model consumes
    its own output, which is why the intervals widen with horizon
    without being told to.
    """
    beta = fit["beta"]
    p = fit["n_lags"]
    nu = fit["nu"]
    alpha = fit["alpha"]
    H = int(horizon)
    hist = [float(v) / nu for v in k.vec(z_history)][-p:]
    if len(hist) < p:
        raise ValueError("deepar: need at least %d history points, got "
                         "%d" % (p, len(hist)))
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(int(n_samples)):
        st = list(hist)
        path = []
        for _h in range(H):
            mu_s = beta[0] + sum(beta[j + 1] * st[-j - 1]
                                 for j in range(p))
            mu_s = max(mu_s, 0.0)
            if fit["likelihood"] == "negative-binomial":
                draw = sample_negative_binomial(mu_s * nu, alpha,
                                                rng) / nu
            else:
                draw = mu_s + (alpha / nu) * rng.standard_normal()
            st.append(draw)
            path.append(draw * nu)          # back to the real scale
        paths.append(path)
    return paths


def deepar_forecast(z, horizon, n_lags=2,
                    likelihood="negative-binomial", n_samples=300,
                    quantiles=(0.1, 0.5, 0.9), seed=0):
    """Fit, sample, and read the quantiles off the trajectories."""
    fit = deepar_fit(z, n_lags=n_lags, likelihood=likelihood)
    paths = deepar_sample(fit, z, horizon, n_samples=n_samples,
                          seed=seed)
    H = int(horizon)
    qs = {}
    for q in quantiles:
        if not 0.0 < float(q) < 1.0:
            raise ValueError("deepar: quantiles must be in (0, 1), got "
                             "%r" % (q,))
        qs[float(q)] = [k.quantile7(sorted(pp[h] for pp in paths),
                                    float(q)) for h in range(H)]
    mean = [k.mean([pp[h] for pp in paths]) for h in range(H)]
    width = [qs[max(quantiles)][h] - qs[min(quantiles)][h]
             for h in range(H)]
    return RichResult(payload={
        "estimate": mean, "mean": mean, "quantiles": qs,
        "paths": paths, "width": width, "horizon": H,
        "nu": fit["nu"], "alpha": fit["alpha"],
        "likelihood": likelihood, "n_samples": int(n_samples),
        "method": "DeepAR probabilistic forecast by ancestral "
                  "sampling, Salinas et al. (2020)",
    })


def cheatsheet():
    return ("deepar: emit LIKELIHOOD PARAMETERS per step, not a point. "
            "Negative binomial with Var = mu(1 + mu*alpha) -- alpha IS "
            "the overdispersion and alpha->0 is Poisson. Scale each "
            "series by nu = 1 + mean(z) (the +1 protects an all-zero "
            "series) or the loss is dominated by the largest series. "
            "Forecast by ancestral SAMPLING; that is why intervals "
            "widen with horizon.")


# compact alias per ledger/NAMING.md
deeparforecast = deepar_forecast

# public names resolved by fn/_lazy_map.json
deepar = deepar_forecast
