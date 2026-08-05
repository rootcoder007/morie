# morie.fn -- wave2 slice x_4_01 (rootcoder007/morie)
"""Privacy amplification by shuffling: local DP to central DP."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["shuffle_model"]


def _exp(v):
    """exp that saturates to +inf instead of raising, matching R's exp()."""
    try:
        return math.exp(v)
    except OverflowError:
        return float("inf")



def shuffle_model(epsilon0, n, delta):
    r"""Central (:math:`\varepsilon, \delta`)-DP guarantee obtained by
    shuffling *n* reports, each produced by an
    :math:`\varepsilon_0`-locally-differentially-private randomiser.

    Erlingsson et al., Theorem 7.  For any integer :math:`n > 1` and
    :math:`\varepsilon_0, \delta > 0`, the shuffled mechanism satisfies
    :math:`(\varepsilon, \delta)`-DP in the central model with

    .. math::

        \varepsilon \le \varepsilon_1 \sqrt{2 n \log(1/\delta)}
                        + n \varepsilon_1 (e^{\varepsilon_1} - 1),
        \qquad
        \varepsilon_1 = \frac{2 e^{2\varepsilon_0}(e^{\varepsilon_0} - 1)}{n}.

    In particular, for :math:`\varepsilon_0 \le \ln(n/4)/3`,

    .. math::

        \varepsilon \le e^{2\varepsilon_0}(e^{\varepsilon_0} - 1)
                        \sqrt{\frac{8 \log(1/\delta)}{n}}
                      + \frac{6 e^{4\varepsilon_0}(e^{\varepsilon_0} - 1)^2}{n},

    and for any :math:`n \ge 1000`, :math:`0 < \varepsilon_0 < 1/2` and
    :math:`0 < \delta < 1/100`, the simple form

    .. math::  \varepsilon \le 12 \varepsilon_0 \sqrt{\log(1/\delta)/n},

    which is the :math:`\tilde\Theta(\sqrt n)` amplification the paper
    advertises.  All three are returned; ``estimate`` is the general bound,
    the only one whose side conditions always hold.  ``simple_valid`` and
    ``refined_valid`` say whether each specialised form applies to the
    arguments given.  Note that for :math:`\varepsilon_0 > \ln(n/2)/3` the
    bound exceeds :math:`\varepsilon_0` and there is no amplification at
    all; ``amplifies`` flags that case.

    Equations read from a rendered image of p. 11 of arXiv:1811.12469v2:
    the text layer drops the square-root signs.

    Parameters
    ----------
    epsilon0 : float
        Per-report local DP parameter, must be positive.
    n : int
        Number of reports shuffled, must exceed 1.
    delta : float
        Target failure probability, in (0, 1).

    Returns
    -------
    RichResult
        ``estimate`` is the general Theorem 7 bound on epsilon.

    References
    ----------
    Erlingsson, U., Feldman, V., Mironov, I., Raghunathan, A., Talwar, K. &
    Thakurta, A. (2019). Amplification by shuffling: from local to central
    differential privacy via anonymity. Proceedings of the Thirtieth Annual
    ACM-SIAM Symposium on Discrete Algorithms, 2468-2479, Theorem 7.
    doi:10.1137/1.9781611975482.151
    """
    e0 = float(epsilon0)
    nn = float(n)
    d = float(delta)
    if e0 <= 0.0:
        raise ValueError("shuffle_model: epsilon0 must be positive")
    if nn <= 1.0:
        raise ValueError("shuffle_model: n must exceed 1")
    if d <= 0.0 or d >= 1.0:
        raise ValueError("shuffle_model: delta must lie in (0, 1)")

    logd = math.log(1.0 / d)
    e1 = 2.0 * _exp(2.0 * e0) * (_exp(e0) - 1.0) / nn
    general = e1 * math.sqrt(2.0 * nn * logd) + nn * e1 * (_exp(e1) - 1.0)

    refined = _exp(2.0 * e0) * (_exp(e0) - 1.0) * math.sqrt(8.0 * logd / nn) \
        + 6.0 * _exp(4.0 * e0) * (_exp(e0) - 1.0) ** 2 / nn
    simple = 12.0 * e0 * math.sqrt(logd / nn)

    refined_valid = 1.0 if e0 <= math.log(nn / 4.0) / 3.0 else 0.0
    simple_valid = 1.0 if (nn >= 1000.0 and e0 < 0.5 and d < 0.01) else 0.0
    amplifies = 1.0 if e0 <= math.log(nn / 2.0) / 3.0 else 0.0

    return RichResult(
        payload={
            "estimate": general,
            "epsilon": general,
            "epsilon_general": general,
            "epsilon_refined": refined,
            "epsilon_simple": simple,
            "epsilon1": e1,
            "epsilon0": e0,
            "delta": d,
            "n": nn,
            "log_inv_delta": logd,
            "amplification_factor": e0 / general if general > 0.0 else float("inf"),
            "refined_valid": refined_valid,
            "simple_valid": simple_valid,
            "amplifies": amplifies,
            "method": "Privacy amplification by shuffling (Erlingsson et al. 2019, Thm 7)",
        }
    )


def cheatsheet():
    return "shfflm: shuffle-model amplification of local DP to central DP"


# compact alias per ledger/NAMING.md
shufflemodel = shuffle_model
