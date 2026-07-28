# morie.fn -- function file (rootcoder007/morie)
"""M-estimator Taylor expansion."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_m_estimator_taylor_expansion"]


def kosorok_ch2_m_estimator_taylor_expansion(m, theta, theta_0, X, m_dot=None):
    r"""Second-order smoothness condition for M-estimators:

    .. math:: P\big[m_\theta - m_{\theta_0}
              - (\theta - \theta_0)' \dot m_{\theta_0}\big]
              = o(\|\theta - \theta_0\|^2).

    This is a condition on the CRITERION's population expectation, not
    on individual sample paths -- m itself may be non-smooth (absolute
    loss is the standard example) while its expectation is twice
    differentiable. Conflating the two is why least-absolute-deviation
    estimators look intractable until this distinction is drawn.

    Returns the remainder along the supplied theta sequence together
    with its ratio to :math:`\|\theta - \theta_0\|^2`, which the
    condition requires to vanish.

    Parameters
    ----------
    m : callable
        m(theta, X) -> per-observation criterion values.
    theta : sequence of array-like
        Parameter values approaching theta_0.
    theta_0 : array-like
        The centre.
    X : array-like
        Sample, used for the empirical expectation.
    m_dot : callable, optional
        m_dot(theta_0, X) -> per-observation gradient; numerical if
        omitted.

    Returns
    -------
    RichResult
        keys: ``distances``, ``remainders``, ``ratios``,
        ``ratio_shrinking``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (M-estimator rates of convergence).
    """
    t0 = np.atleast_1d(np.asarray(theta_0, dtype=float))
    X = np.asarray(X, dtype=float)
    base = float(np.mean(m(t0, X)))
    if m_dot is None:
        h = 1e-6
        grad = np.array([
            (float(np.mean(m(t0 + h * e, X))) - float(np.mean(m(t0 - h * e, X))))
            / (2 * h)
            for e in np.eye(t0.size)
        ])
    else:
        grad = np.atleast_1d(np.mean(np.atleast_2d(m_dot(t0, X)), axis=0))
    seq = [np.atleast_1d(np.asarray(t, dtype=float)) for t in theta]
    if len(seq) < 2:
        raise ValueError("theta must contain at least 2 values.")
    d, rem = [], []
    for t in seq:
        delta = t - t0
        nrm = float(np.linalg.norm(delta))
        if nrm == 0:
            raise ValueError("theta values must differ from theta_0.")
        d.append(nrm)
        rem.append(abs(float(np.mean(m(t, X))) - base - float(delta @ grad)))
    d = np.array(d)
    rem = np.array(rem)
    ratios = rem / d**2
    order = np.argsort(-d)
    return RichResult(
        payload={"distances": d, "remainders": rem, "ratios": ratios,
                 "ratio_shrinking": bool(ratios[order][-1] <= ratios[order][0] * 1.5),
                 "method": "P[m_theta - m_0 - delta' m_dot] / ||delta||^2 (Kosorok Ch. 2)"}
    )


def cheatsheet():
    return "ksr055: smoothness of the EXPECTATION, not of m itself"
