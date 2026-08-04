"""Quasi-likelihood estimation of the SV(1) stochastic volatility model."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["vol_sv_quasi_lik"]

# log of a chi-square(1) variate: mean psi(1/2) + log 2 = -gamma - log 2,
# variance psi'(1/2) = pi ** 2 / 2.  These are the two constants that make
# the linearised measurement equation a valid quasi-likelihood.
_LOGCHI2_MEAN = -1.2703628454614782
_LOGCHI2_VAR = 4.934802200544679

_GOLD = 0.3819660112501051  # (3 - sqrt(5)) / 2


def _kalman_qll(y, mu, phi, sig2):
    """Gaussian quasi log-likelihood from the Kalman filter."""
    n = len(y)
    if not (-0.999999 < phi < 0.999999) or sig2 <= 0.0:
        return -1e300
    a = 0.0
    p = sig2 / (1.0 - phi * phi)      # stationary prior on h*
    c = mu + _LOGCHI2_MEAN
    ll = 0.0
    for t in range(n):
        v = y[t] - c - a              # innovation
        f = p + _LOGCHI2_VAR          # innovation variance
        if f <= 0.0:
            return -1e300
        ll += -0.5 * (np.log(2.0 * 3.141592653589793 * f) + v * v / f)
        k = p / f
        a = a + k * v
        p = p - k * p
        a = phi * a                   # transition
        p = phi * phi * p + sig2
    return ll


def _golden(f, lo, hi, iters=80):
    """Deterministic golden-section maximisation on [lo, hi]."""
    x1 = lo + _GOLD * (hi - lo)
    x2 = hi - _GOLD * (hi - lo)
    f1 = f(x1)
    f2 = f(x2)
    for _ in range(iters):
        if f1 < f2:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = hi - _GOLD * (hi - lo)
            f2 = f(x2)
        else:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = lo + _GOLD * (hi - lo)
            f1 = f(x1)
    return x1 if f1 >= f2 else x2


def vol_sv_quasi_lik(r, init=None, sweeps=25, offset=1e-8):
    """
    Quasi-likelihood SV(1) via the Kalman filter

    Formula: log r_t ** 2 = h_t + log z_t ** 2, h_t = mu + phi (h_{t-1} - mu) + eta_t

    The stochastic volatility model ``r_t = exp(h_t / 2) z_t`` with
    ``z_t`` standard normal and ``h_t`` a Gaussian AR(1) is not a linear
    Gaussian state space model, so the exact likelihood needs
    simulation.  Squaring and taking logs linearises it,

        log r_t ** 2 = h_t + log z_t ** 2,

    at the price of a measurement error ``log z_t ** 2`` that is log of a
    chi-square with one degree of freedom, and so is very far from
    normal: it is sharply left-skewed.  Treating it as normal anyway,
    with its true first two moments

        E[log z ** 2]   = psi(1/2) + log 2 = -gamma - log 2 = -1.2703628
        Var[log z ** 2] = psi'(1/2) = pi ** 2 / 2 = 4.9348022,

    turns the model into a linear Gaussian one whose Kalman filter
    prediction errors give a *quasi* likelihood.  Maximising it is
    consistent and asymptotically normal but not efficient, and the
    reported ``ll`` is a quasi log-likelihood, not a log-likelihood: it
    is not comparable with the likelihood of a model estimated exactly.

    The filter is run on the state ``h*_t = h_t - mu``, with the
    stationary prior ``Var(h*_0) = sigma_eta ** 2 / (1 - phi ** 2)`` and
    the measurement intercept ``mu + E[log z ** 2]``.

    Maximisation is a deterministic coordinate ascent: golden-section
    line search on each of ``mu``, ``phi`` and ``log sigma_eta`` in turn,
    for a fixed number of sweeps.  There is no random restart and no
    convergence tolerance, so the result is a deterministic function of
    the data and of ``init``.

    Zero returns make ``log r_t ** 2`` infinite.  ``offset`` is added to
    ``r_t ** 2`` before the log to keep the filter finite; it is the
    usual remedy and it biases the estimate slightly, so it is reported.

    Parameters
    ----------
    r : array-like
        Returns, not squared and not logged.
    init : sequence of 3 floats, optional
        Starting ``(mu, phi, sigma_eta)``.  Defaults to a method of
        moments start read off ``log r_t ** 2``.
    sweeps : int
        Number of coordinate-ascent sweeps.
    offset : float
        Added to ``r_t ** 2`` before taking logs.

    Returns
    -------
    result : RichResult
        Keys: mu, phi, sigma_eta, ll, n, sweeps, offset, init, method.

    References
    ----------
    Harvey A C, Ruiz E & Shephard N (1994).  Multivariate stochastic
    variance models.  Review of Economic Studies 61(2), 247-264.  The
    linearisation above and the quasi-likelihood built from the Kalman
    filter prediction errors are that paper's estimator, specialised to
    one series.
    """
    rv = [float(v) for v in np.atleast_1d(np.asarray(r, dtype=float)).tolist()]
    n = len(rv)
    if n < 10:
        raise ValueError("need at least ten observations")
    offset = float(offset)
    if offset < 0.0:
        raise ValueError("offset must be non-negative")
    y = [np.log(v * v + offset) for v in rv]

    if init is None:
        ybar = 0.0
        for v in y:
            ybar += v
        ybar /= n
        # Var(y) = Var(h) + pi^2/2, so a moment start for sigma_eta.
        s2 = 0.0
        for v in y:
            s2 += (v - ybar) ** 2
        s2 /= (n - 1)
        vh = s2 - _LOGCHI2_VAR
        if vh <= 0.0:
            vh = 0.05
        phi0 = 0.9
        init = (ybar - _LOGCHI2_MEAN, phi0, np.sqrt(vh * (1.0 - phi0 * phi0)))
    iv = [float(v) for v in init]
    if len(iv) != 3:
        raise ValueError("init must be (mu, phi, sigma_eta)")
    mu, phi, sig = iv
    if not (-0.999999 < phi < 0.999999):
        raise ValueError("initial phi must lie strictly inside (-1, 1)")
    if sig <= 0.0:
        raise ValueError("initial sigma_eta must be positive")
    lsig = np.log(sig)

    sweeps = int(sweeps)
    if sweeps < 1:
        raise ValueError("sweeps must be at least 1")
    for _s in range(sweeps):
        mu = _golden(lambda v: _kalman_qll(y, v, phi, np.exp(2.0 * lsig)),
                     mu - 5.0, mu + 5.0)
        phi = _golden(lambda v: _kalman_qll(y, mu, v, np.exp(2.0 * lsig)),
                      -0.999, 0.999)
        lsig = _golden(lambda v: _kalman_qll(y, mu, phi, np.exp(2.0 * v)),
                       lsig - 3.0, lsig + 3.0)

    sig = np.exp(lsig)
    ll = _kalman_qll(y, mu, phi, sig * sig)
    return RichResult(
        payload={
            "mu": float(mu),
            "phi": float(phi),
            "sigma_eta": float(sig),
            "ll": float(ll),
            "n": n,
            "sweeps": sweeps,
            "offset": offset,
            "init": [float(v) for v in iv],
            "method": "SV(1) quasi-likelihood via Kalman filter (Harvey-Ruiz-Shephard 1994)",
        }
    )


def cheatsheet():
    return "volsv: SV(1) quasi-likelihood via the Kalman filter"


# compact alias per ledger/NAMING.md
svquasilik = vol_sv_quasi_lik
