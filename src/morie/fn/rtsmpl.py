# morie.fn -- function file (rootcoder007/morie)
"""Instantaneous reproduction number over a sliding window.

Source CONSULTED via its reference implementation: Cori, A., Ferguson,
N. M., Fraser, C. & Cauchemez, S. (2013), "A new framework and software
to estimate time-varying reproduction numbers during epidemics",
*American Journal of Epidemiology* 178:1505-1512.  The AJE article
itself is paywalled, so the exact estimator was taken from the authors'
own EpiEstim package (mrc-ide/EpiEstim), which the paper distributes as
its software.  Quoted verbatim from ``R/overall_infectivity.R``:

    lambda[t] <- sum(si_distr[seq_len(t)] *
                     rowSums(incid[seq(t, 1), c("local", "imported")]),
                     na.rm = TRUE)

and from ``R/estimate_R.R``:

    a_posterior <- a_prior + sum(incid[seq(t_start[t], t_end[t]), "local"])
    b_posterior <- 1 / (1 / b_prior + sum(lambda[seq(t_start[t], t_end[t])]))
    mean_posterior <- a_posterior * b_posterior
    std_posterior  <- sqrt(a_posterior) * b_posterior

The package default prior has mean 5 and standard deviation 5, i.e. the
gamma shape a = 1 and scale b = 5 used here.
"""

import math

from ._richresult import RichResult

__all__ = ["rt_serial_interval"]


def rt_serial_interval(incidence, serial_interval, window=7,
                       a_prior=1.0, b_prior=5.0):
    """Cori et al. (2013) instantaneous reproduction number.

    Parameters
    ----------
    incidence : sequence
        Daily incidence I_0, I_1, ..., I_{T-1}.
    serial_interval : sequence
        Discrete serial-interval distribution w_0, w_1, ...  By the
        EpiEstim convention w_0 = 0; the vector is used as supplied.
    window : int
        Length tau of the sliding estimation window, in days.
    a_prior, b_prior : float
        Shape and SCALE of the gamma prior on R.  The package default
        prior (mean 5, sd 5) is a = 1, b = 5.

    Returns
    -------
    RichResult
        ``r_mean``, ``r_std``, ``a_posterior``, ``b_posterior``,
        ``lambda`` (total infectiousness, index 0 undefined and
        reported as 0.0), ``t_start``, ``t_end``, ``n``.
    """
    inc = [float(v) for v in incidence]
    w = [float(v) for v in serial_interval]
    T = len(inc)
    tau = int(window)
    if T == 0:
        raise ValueError("incidence is empty")
    if tau < 1 or tau > T:
        raise ValueError("window must lie in 1..len(incidence)")
    if not w:
        raise ValueError("serial_interval is empty")

    # overall_infectivity: lambda[t] = sum_k w_k I_{t-k}.  R indexes
    # si_distr from 1 and starts the loop at t = 2, so the 0-based
    # equivalent starts at t = 1 and lambda[0] is undefined.
    lam = [0.0] * T
    for t in range(1, T):
        acc = 0.0
        for k in range(t + 1):
            if k < len(w):
                acc += w[k] * inc[t - k]
        lam[t] = acc

    # Sliding windows [t_start, t_end] inclusive, 0-based, the first
    # window ending at index tau (EpiEstim skips the first day, whose
    # lambda is undefined).
    t_start = []
    t_end = []
    a_post = []
    b_post = []
    r_mean = []
    r_std = []
    for end in range(tau, T):
        start = end - tau + 1
        a = a_prior + sum(inc[start:end + 1])
        b = 1.0 / (1.0 / b_prior + sum(lam[start:end + 1]))
        t_start.append(start)
        t_end.append(end)
        a_post.append(float(a))
        b_post.append(float(b))
        r_mean.append(float(a * b))
        r_std.append(float(math.sqrt(a) * b))

    return RichResult(payload={
        "r_mean": r_mean, "r_std": r_std,
        "a_posterior": a_post, "b_posterior": b_post,
        "lambda": [float(v) for v in lam],
        "t_start": t_start, "t_end": t_end,
        "window": tau, "a_prior": float(a_prior), "b_prior": float(b_prior),
        "n_windows": len(r_mean), "n": T,
        "method": "Cori et al. (2013) instantaneous R, gamma-Poisson conjugate posterior"})


def cheatsheet():
    return "rtsmpl: Cori et al. (2013) instantaneous reproduction number"


# compact alias per ledger/NAMING.md
rtsi = rt_serial_interval
