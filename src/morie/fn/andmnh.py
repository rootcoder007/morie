# morie.fn -- function file (rootcoder007/morie)
r"""VAR prewhitened kernel HAC covariance matrix estimation.

Andrews, D. W. K., & Monahan, J. C. (1992) "An Improved
Heteroskedasticity and Autocorrelation Consistent Covariance Matrix
Estimator", *Econometrica* 60(4), 953-966. doi:10.2307/2951574

The kernel constants, the automatic bandwidth, and the
:math:`\alpha(q)` formulae are Andrews (1991), used here through the
Cowles Foundation Discussion Paper 877R of July 1989, "Heteroskedasticity
and Autocorrelation Consistent Covariance Matrix Estimation" -- the
working version of *Econometrica* 59(3), 817-858.

A kernel HAC estimator averages sample autocovariances over a
neighbourhood of frequency zero. Averaging is only unbiased where the
spectral density is flat, so the more temporal dependence there is, the
worse the bias -- which is exactly the regime where a HAC estimator is
needed. Andrews & Monahan's answer is the old prewhitening idea of
Press & Tukey (1956): filter first, so that what the kernel sees is
closer to white noise, then undo the filter on the estimate.

**Step one, equation 2.2.** Fit a VAR of order :math:`b` to
:math:`V_t(\hat\theta)`,

.. math::

   V_t(\hat\theta) = \sum_{r=1}^{b} \hat{A}_r V_{t-r}(\hat\theta)
                     + \hat{V}^{*}_t, \qquad t = b+1, \ldots, T.

The VAR is not a model of anything; it is a sponge for temporal
dependence. Section 3 uses :math:`b = 1` and a least-squares
:math:`\hat{A}` adjusted through its singular value decomposition:
write :math:`\hat{A}_{LS} = B \Lambda C'` with :math:`B` and :math:`C`
orthogonal, replace any element of :math:`\Lambda` above
:math:`0.97` by :math:`0.97` (and below :math:`-0.97` by
:math:`-0.97`), and set :math:`\hat{A} = B \bar\Lambda C'`. Footnote 4
proves this keeps every eigenvalue of :math:`I_p - \hat{A}` at least
:math:`0.03` from zero, so the recolouring below cannot blow up. It
costs nothing asymptotically.

**Step two, equation 2.3.** A standard kernel estimator on the
residuals,

.. math::

   \hat{J}^{*}_T(S_T) = \frac{T}{T-\ell}
     \sum_{j=-(T-1)}^{T-1} k\!\left(\frac{j}{S_T}\right) \Gamma^{*}(j),
   \qquad
   \Gamma^{*}(j) = \frac{1}{T} \sum_{t=j+1}^{T}
     \hat{V}^{*}_t \hat{V}^{*\prime}_{t-j}

for :math:`j \geq 0` and :math:`\Gamma^{*}(j) = \Gamma^{*}(-j)'`
otherwise. The factor :math:`T/(T-\ell)` is a degrees-of-freedom
correction for having estimated the :math:`\ell`-vector
:math:`\theta_0`.

**Step three, equation 2.4.** Recolour:

.. math::

   \hat{J}^{pw}_T(S_T) = \hat{D}\, \hat{J}^{*}_T(S_T)\, \hat{D}',
   \qquad \hat{D} = \Bigl(I_p - \sum_{r=1}^{b} \hat{A}_r\Bigr)^{-1}.

Setting :math:`\hat{A} = 0` turns this back into the ordinary kernel
estimator, which is the paper's own QS comparison estimator, so
``prewhiten=False`` is a route rather than a degradation.

**The bandwidth.** Andrews (1991) eq. 6.1,

.. math::

   \hat{S}_T = \left( q\, k_q^2\, \hat\alpha(q)\, T \Big/
                      \int k^2(x)\,dx \right)^{1/(2q+1)},

with :math:`q` the characteristic exponent of the kernel and
:math:`k_q = \lim_{x \to 0}(1 - k(x))/|x|^q`. For the quadratic
spectral kernel :math:`q = 2`, :math:`k_q = 1.421223` and
:math:`\int k^2 = 1`, which collapses to the paper's equation 3.5,
:math:`\hat{S}_T = 1.3221(\hat\alpha(2) T)^{1/5}`.
:math:`\hat\alpha(q)` comes from :math:`p` univariate AR(1)
approximating models (Andrews 1991 eq. 6.4),

.. math::

   \hat\alpha(2) = \frac{\sum_a w_a\, 4\hat\rho_a^2 \hat\sigma_a^4 /
                          (1-\hat\rho_a)^8}
                        {\sum_a w_a\, \hat\sigma_a^4 /
                          (1-\hat\rho_a)^4},
   \quad
   \hat\alpha(1) = \frac{\sum_a w_a\, 4\hat\rho_a^2 \hat\sigma_a^4 /
                          \{(1-\hat\rho_a)^6 (1+\hat\rho_a)^2\}}
                        {\sum_a w_a\, \hat\sigma_a^4 /
                          (1-\hat\rho_a)^4}.

Section 3 gives no weight to the intercept, :math:`w = (0, 1, \ldots,
1)`; that is ``weights="drop_first"`` here, and the general default is
equal weights.

Four kernels are available. Their :math:`(q, k_q, \int k^2)` are
tabulated in :data:`KERNEL_CONSTANTS` and every one of them is
recomputed from the kernel function itself in the tests, so nothing
here rests on a transcribed number.
"""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "andmnh",
    "andrews_monahan_hac",
    "moment_vectors",
    "prewhiten_var",
    "singular_value_adjust",
    "kernel_hac",
    "automatic_bandwidth",
    "alpha_ar1",
    "ar1_fit",
    "bartlett_kernel",
    "parzen_kernel",
    "quadratic_spectral_kernel",
    "tukey_hanning_kernel",
    "KERNELS",
    "KERNEL_CONSTANTS",
]

#: Singular values of the prewhitening matrix are capped here (Section 3).
EIGENVALUE_CAP = 0.97


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------

def bartlett_kernel(x):
    """Newey & West's triangular kernel; :math:`q = 1`."""
    ax = abs(float(x))
    return 1.0 - ax if ax <= 1.0 else 0.0


def parzen_kernel(x):
    """Parzen's kernel; :math:`q = 2`."""
    ax = abs(float(x))
    if ax <= 0.5:
        return 1.0 - 6.0 * ax * ax + 6.0 * ax ** 3
    if ax <= 1.0:
        return 2.0 * (1.0 - ax) ** 3
    return 0.0


def quadratic_spectral_kernel(x):
    r"""Equation 3.2, the QS kernel.

    .. math::

       k_{QS}(x) = \frac{25}{12\pi^2 x^2}
         \left\{ \frac{\sin(6\pi x/5)}{6\pi x/5} - \cos(6\pi x/5)
         \right\}

    It has unbounded support, so every lag contributes, and it makes
    the recoloured estimator positive semi-definite.
    """
    x = float(x)
    if x == 0.0:
        return 1.0
    z = 6.0 * math.pi * x / 5.0
    return (25.0 / (12.0 * math.pi ** 2 * x * x)) * (math.sin(z) / z
                                                     - math.cos(z))


def tukey_hanning_kernel(x):
    """The Tukey-Hanning kernel; :math:`q = 2`."""
    ax = abs(float(x))
    return 0.5 * (1.0 + math.cos(math.pi * ax)) if ax <= 1.0 else 0.0


KERNELS = {
    "bartlett": bartlett_kernel,
    "parzen": parzen_kernel,
    "qs": quadratic_spectral_kernel,
    "tukey-hanning": tukey_hanning_kernel,
}

#: ``name -> (q, k_q, integral of k^2, has bounded support)``.
#: ``q`` is the largest exponent with :math:`\lim (1-k(x))/|x|^q`
#: finite and non-zero, and ``k_q`` is that limit.
KERNEL_CONSTANTS = {
    "bartlett": (1, 1.0, 2.0 / 3.0, True),
    "parzen": (2, 6.0, 0.539285, True),
    "qs": (2, 1.421223, 1.0, False),
    "tukey-hanning": (2, math.pi ** 2 / 4.0, 0.75, True),
}


def _check_kernel(kernel):
    if kernel not in KERNELS:
        raise ValueError("andmnh: kernel must be one of %s, got %r"
                         % (sorted(KERNELS), kernel))
    return KERNELS[kernel], KERNEL_CONSTANTS[kernel]


# --------------------------------------------------------------------------
# moment vectors and the prewhitening VAR
# --------------------------------------------------------------------------

def moment_vectors(e, X):
    r"""Section 3: :math:`V_t(\hat\theta) = X_t (Y_t - X_t'\hat\theta)`.

    For least squares this is the regressor row scaled by the residual,
    which is the score whose long-run variance the sandwich needs.
    """
    e = [float(v) for v in e]
    rows = [[float(v) for v in row] for row in X]
    if len(rows) != len(e):
        raise ValueError("andmnh: %d residuals but %d regressor rows"
                         % (len(e), len(rows)))
    if not rows:
        raise ValueError("andmnh: no observations")
    p = len(rows[0])
    for row in rows:
        if len(row) != p:
            raise ValueError("andmnh: ragged regressor matrix")
    return [[rows[t][j] * e[t] for j in range(p)] for t in range(len(e))]


def singular_value_adjust(a, cap=EIGENVALUE_CAP):
    r"""Section 3's SVD adjustment of the prewhitening matrix.

    :math:`\hat{A}_{LS} = B \Lambda C'` with :math:`B`, :math:`C`
    orthogonal; elements of :math:`\Lambda` outside
    :math:`[-cap, cap]` are pulled to the boundary and
    :math:`\hat{A} = B \bar\Lambda C'`. Footnote 4: this forces every
    eigenvalue of :math:`I_p - \hat{A}` at least :math:`1 - cap` from
    zero, since :math:`|\lambda| \leq cap` for every eigenvalue
    :math:`\lambda` of :math:`\hat{A}`.
    """
    cap = float(cap)
    if not 0.0 < cap < 1.0:
        raise ValueError("andmnh: cap must lie strictly between 0 and 1")
    am = np.asarray(a, dtype=float)
    u, s, vt = np.linalg.svd(am)
    s2 = [min(float(v), cap) for v in s]     # singular values are >= 0
    return np.dot(np.dot(np.asarray(u), np.diag(np.asarray(s2))),
                  np.asarray(vt))


def prewhiten_var(v, order=1, cap=EIGENVALUE_CAP, adjust=True):
    r"""Equation 2.2: fit the VAR and return its coefficients and residuals.

    Returns ``(A_list, residuals, D)`` where ``D`` is the recolouring
    matrix :math:`(I_p - \sum_r \hat{A}_r)^{-1}` of equation 2.4.
    ``order=0`` means no prewhitening at all, which is the paper's own
    comparison estimator with :math:`\hat{A} = 0`.
    """
    rows = [[float(x) for x in row] for row in v]
    n = len(rows)
    if not n:
        raise ValueError("andmnh: no observations")
    p = len(rows[0])
    order = int(order)
    if order < 0:
        raise ValueError("andmnh: VAR order must be non-negative")
    if order == 0:
        return [], rows, np.eye(p)
    if n <= order * p + 1:
        raise ValueError(
            "andmnh: %d observations cannot fit a VAR(%d) in %d variables"
            % (n, order, p))

    # least squares of V_t on V_{t-1}, ..., V_{t-order}
    y = [rows[t] for t in range(order, n)]
    z = [[rows[t - r - 1][j] for r in range(order) for j in range(p)]
         for t in range(order, n)]
    zm = np.asarray(z, dtype=float)
    ym = np.asarray(y, dtype=float)
    coef = np.linalg.lstsq(zm, ym, rcond=None)[0]    # (order*p) x p
    coef = np.asarray(coef)

    a_list = []
    for r in range(order):
        block = [[float(coef[r * p + j][i]) for j in range(p)]
                 for i in range(p)]
        a_list.append(np.asarray(block, dtype=float))

    if adjust:
        a_list = [singular_value_adjust(a, cap) for a in a_list]
        if order > 1:
            # The paper specifies the adjustment for b = 1. For a longer
            # VAR, capping each A_r is not enough to bound the sum, so
            # shrink them together until I - sum(A_r) is as far from
            # singular as the b = 1 case guarantees.
            for _ in range(200):
                tot = a_list[0]
                for a in a_list[1:]:
                    tot = tot + a
                smax = max(float(s) for s in np.linalg.svd(tot)[1])
                if smax <= cap:
                    break
                a_list = [a * (cap / smax) for a in a_list]

    resid = []
    for t in range(order, n):
        pred = [0.0] * p
        for r in range(order):
            ar = a_list[r]
            for i in range(p):
                pred[i] += sum(float(ar[i][j]) * rows[t - r - 1][j]
                               for j in range(p))
        resid.append([rows[t][i] - pred[i] for i in range(p)])

    tot = a_list[0]
    for a in a_list[1:]:
        tot = tot + a
    d = np.linalg.inv(np.eye(p) - np.asarray(tot))
    return a_list, resid, d


# --------------------------------------------------------------------------
# the bandwidth
# --------------------------------------------------------------------------

def ar1_fit(x):
    r"""Least squares AR(1) without an intercept: :math:`(\rho, \sigma^2)`."""
    x = [float(v) for v in x]
    n = len(x)
    if n < 3:
        raise ValueError("andmnh: an AR(1) needs at least 3 observations")
    num = sum(x[t] * x[t - 1] for t in range(1, n))
    den = sum(x[t - 1] * x[t - 1] for t in range(1, n))
    rho = num / den if den > 0.0 else 0.0
    s2 = sum((x[t] - rho * x[t - 1]) ** 2 for t in range(1, n)) / (n - 1)
    return rho, s2


def alpha_ar1(v, q=2, weights=None):
    r"""Andrews (1991) eq. 6.4: :math:`\hat\alpha(q)` from AR(1) models.

    ``weights`` is :math:`(w_1, \ldots, w_p)`; ``"drop_first"`` gives
    the Section 3 choice :math:`(0, 1, \ldots, 1)`, which withholds
    weight from the intercept.
    """
    rows = [[float(x) for x in row] for row in v]
    if not rows:
        raise ValueError("andmnh: no observations")
    p = len(rows[0])
    if weights is None:
        w = [1.0] * p
    elif weights == "drop_first":
        w = [0.0] + [1.0] * (p - 1)
    else:
        w = [float(x) for x in weights]
        if len(w) != p:
            raise ValueError("andmnh: %d weights for %d series"
                             % (len(w), p))
    if any(x < 0.0 for x in w) or sum(w) <= 0.0:
        raise ValueError("andmnh: weights must be non-negative and not "
                         "all zero")
    q = int(q)
    if q not in (1, 2):
        raise ValueError("andmnh: alpha(q) is given for q = 1 or 2")

    num = 0.0
    den = 0.0
    fits = []
    for a in range(p):
        rho, s2 = ar1_fit([row[a] for row in rows])
        fits.append((rho, s2))
        if w[a] == 0.0:
            continue
        s4 = s2 * s2
        if q == 2:
            num += w[a] * 4.0 * rho * rho * s4 / (1.0 - rho) ** 8
        else:
            num += (w[a] * 4.0 * rho * rho * s4
                    / ((1.0 - rho) ** 6 * (1.0 + rho) ** 2))
        den += w[a] * s4 / (1.0 - rho) ** 4
    if den <= 0.0:
        raise ValueError("andmnh: the alpha(q) denominator vanished")
    return num / den, fits


def automatic_bandwidth(v, kernel="qs", weights=None, n=None):
    r"""Andrews (1991) eq. 6.1 with the AR(1) plug-in.

    .. math::

       \hat{S}_T = \left(q k_q^2 \hat\alpha(q) T \Big/
                          \int k^2 \right)^{1/(2q+1)}

    For the QS kernel this is the paper's equation 3.5,
    :math:`1.3221(\hat\alpha(2)T)^{1/5}`.
    """
    _, (q, kq, ik2, _) = _check_kernel(kernel)
    rows = list(v)
    t = int(n) if n is not None else len(rows)
    alpha, fits = alpha_ar1(rows, q=q, weights=weights)
    s = (q * kq * kq * alpha * t / ik2) ** (1.0 / (2.0 * q + 1.0))
    return s, alpha, fits


# --------------------------------------------------------------------------
# the kernel estimator and the whole thing
# --------------------------------------------------------------------------

def kernel_hac(v, bandwidth, kernel="qs", n_params=0, n=None):
    r"""Equation 2.3 on already-prewhitened vectors.

    ``n`` is the :math:`T` used as the divisor of :math:`\Gamma^{*}(j)`
    and in the :math:`T/(T-\ell)` correction; it defaults to the number
    of rows supplied, and should be the full sample size when the rows
    are VAR residuals (which are :math:`T - b` in number).
    """
    kfun, (_, _, _, bounded) = _check_kernel(kernel)
    rows = [[float(x) for x in row] for row in v]
    m = len(rows)
    if not m:
        raise ValueError("andmnh: no observations")
    p = len(rows[0])
    t = int(n) if n is not None else m
    if t <= n_params:
        raise ValueError("andmnh: T = %d is not larger than the %d "
                         "estimated parameters" % (t, n_params))
    s = float(bandwidth)
    if s <= 0.0:
        raise ValueError("andmnh: bandwidth must be positive")

    jmax = m - 1
    if bounded:
        jmax = min(jmax, int(math.floor(s)))

    out = [[0.0] * p for _ in range(p)]
    for j in range(0, jmax + 1):
        kj = kfun(j / s)
        if kj == 0.0:
            continue
        gam = [[0.0] * p for _ in range(p)]
        for tt in range(j, m):
            a = rows[tt]
            b = rows[tt - j]
            for i in range(p):
                ai = a[i]
                if ai == 0.0:
                    continue
                for k in range(p):
                    gam[i][k] += ai * b[k]
        for i in range(p):
            for k in range(p):
                gam[i][k] /= t
        if j == 0:
            for i in range(p):
                for k in range(p):
                    out[i][k] += kj * gam[i][k]
        else:
            # Gamma(-j) = Gamma(j)', and k is even
            for i in range(p):
                for k in range(p):
                    out[i][k] += kj * (gam[i][k] + gam[k][i])

    dof = t / float(t - n_params)
    return np.asarray([[dof * out[i][k] for k in range(p)]
                       for i in range(p)], dtype=float)


def andrews_monahan_hac(e, X=None, prewhiten=True, var_order=1,
                        kernel="qs", bandwidth=None, weights=None,
                        n_params=None, cap=EIGENVALUE_CAP, adjust=True):
    r"""The VAR prewhitened kernel HAC estimator of equation 2.4.

    Parameters
    ----------
    e : array-like
        Either the :math:`T \times p` matrix of moment vectors
        :math:`V_t(\hat\theta)`, or -- when ``X`` is given -- the
        length-:math:`T` residual vector.
    X : array-like, optional
        The :math:`T \times p` regressor matrix. When present the
        moment vectors are formed as :math:`V_t = X_t e_t`, which is
        the Section 3 construction for least squares.
    prewhiten : bool
        ``False`` sets :math:`\hat{A} = 0`, giving the paper's
        unprewhitened comparison estimator through the same code.
    var_order : int
        :math:`b` in equation 2.2. Section 3 uses 1.
    kernel : {"qs", "bartlett", "parzen", "tukey-hanning"}
        Section 3 uses the QS kernel, which alone among these makes
        :math:`\hat{J}^{pw}_T` positive semi-definite.
    bandwidth : float, optional
        A fixed :math:`S_T`. Omitted, the automatic plug-in of
        Andrews (1991) eq. 6.1 is used on the prewhitened residuals.
    weights : sequence or "drop_first", optional
        The :math:`w_a` of :math:`\hat\alpha(q)`. Section 3 uses
        ``"drop_first"`` so the intercept gets no weight.
    n_params : int, optional
        :math:`\ell`, for the :math:`T/(T-\ell)` correction. Defaults
        to :math:`p` when ``X`` is given and to 0 otherwise.

    Returns
    -------
    RichResult
        ``J`` (the recoloured estimator), ``J_star`` (before
        recolouring), ``A`` (the VAR coefficients), ``D``, ``bandwidth``,
        ``alpha``, and the AR(1) fits behind the bandwidth.
    """
    if X is not None:
        v = moment_vectors(e, X)
        if n_params is None:
            n_params = len(v[0])
    else:
        v = [[float(x) for x in row] for row in e]
        if n_params is None:
            n_params = 0
    n = len(v)
    if not n:
        raise ValueError("andmnh: no observations")
    p = len(v[0])
    order = int(var_order) if prewhiten else 0

    a_list, resid, d = prewhiten_var(v, order=order, cap=cap, adjust=adjust)

    if bandwidth is None:
        s, alpha, fits = automatic_bandwidth(resid, kernel=kernel,
                                             weights=weights, n=n)
        auto = True
    else:
        s, alpha, fits = float(bandwidth), None, None
        auto = False

    jstar = kernel_hac(resid, s, kernel=kernel, n_params=n_params, n=n)
    dm = np.asarray(d)
    j = np.dot(np.dot(dm, np.asarray(jstar)), np.asarray(dm).T)

    return RichResult(payload={
        "J": j,
        "J_star": jstar,
        "D": dm,
        "A": a_list,
        "bandwidth": float(s),
        "bandwidth_automatic": auto,
        "alpha": alpha,
        "ar1_fits": fits,
        "kernel": kernel,
        "var_order": order,
        "n": n,
        "p": p,
        "n_params": int(n_params),
        "prewhitened": bool(order),
        "method": ("Andrews & Monahan (1992) VAR prewhitened kernel HAC, "
                   "eq. 2.2-2.4, with the Andrews (1991) eq. 6.1 "
                   "automatic bandwidth"),
        "note": ("the VAR is a filter, not a model; its coefficients are "
                 "capped through their SVD at %.2f so that I - sum(A_r) "
                 "stays %.2f away from singular (footnote 4)"
                 % (cap, 1.0 - cap)),
    })


andmnh = andrews_monahan_hac


def cheatsheet():
    return ("andmnh: Andrews & Monahan (1992) prewhitened HAC. Fit a "
            "VAR(b) to the moment vectors (eq. 2.2), run an ordinary "
            "kernel estimator on its residuals (eq. 2.3), then recolour "
            "with D = (I - sum A_r)^{-1} (eq. 2.4). Prewhitening cuts "
            "bias sharply where dependence is strong, at the price of "
            "variance. The VAR coefficients are capped through their "
            "SVD at 0.97 so the recolouring cannot blow up. Bandwidth "
            "is the Andrews (1991) AR(1) plug-in, which for the QS "
            "kernel is 1.3221 (alpha(2) T)^(1/5). prewhiten=False gives "
            "the unprewhitened kernel estimator; kernel= chooses among "
            "qs, bartlett, parzen and tukey-hanning.")
