# morie.fn -- function file (rootcoder007/morie)
"""Cohen's d standardised mean difference, and Hedges' bias-corrected g.

Sources CONSULTED:

* Cohen, J. (1988), *Statistical Power Analysis for the Behavioral
  Sciences*, 2nd ed., Lawrence Erlbaum.  This is a book and could not be
  obtained in full; the definition implemented is the standard published
  statement of Cohen's d for two independent groups,

      d = (m1 - m2) / s_pooled,
      s_pooled = sqrt( ((n1-1) s1^2 + (n2-1) s2^2) / (n1 + n2 - 2) ).

* Hedges, L. V. (1981), "Distribution theory for Glass's estimator of
  effect size and related estimators", *Journal of Educational
  Statistics* 6(2):107-128, and Hedges & Olkin (1985), *Statistical
  Methods for Meta-Analysis*, Academic Press.  These give the bias
  correction and the large-sample variance used here.  With
  df = n1 + n2 - 2 the exact correction factor is

      J  = Gamma(df/2) / ( sqrt(df/2) * Gamma((df-1)/2) )
      g  = J * d
      var(g) = 1/n1 + 1/n2 + g^2 / (2 (n1 + n2))

  and var(d) is the same expression evaluated at d.  Note that
  1/n1 + 1/n2 = (n1+n2)/(n1 n2), the form Hedges & Olkin print.

TWO CONVENTIONS, settled against the reference implementation.  The
widely quoted approximation J ~= 1 - 3/(4 df - 1) and the alternative
variance var(g) = J^2 var(d) are BOTH in circulation.  Checked against
``metafor::escalc(measure = "SMD")`` -- Viechtbauer's implementation,
the de-facto reference -- for m1=5.4, m2=4.1, s1=1.2, s2=1.5, n1=30,
n2=25.  escalc returns yi = 0.953378008402588 and
vi = 0.0815963299415668.  The exact-Gamma J reproduces yi to 1e-15,
while the 1-3/(4df-1) approximation is out by 1.1e-5; and the variance
evaluated AT g reproduces vi to 1e-15, while J^2 var(d) gives
0.07952603540606365, out by 2.1e-3.  This module therefore uses the
exact J and the variance evaluated at the corrected estimate, and
returns the approximation as ``j_approx`` for reference only.

Only the two-independent-groups case is implemented.  Paired designs use
a different standardiser and are deliberately not folded in here.
"""

import math

from ._richresult import RichResult

__all__ = ["ma_cohens_d"]


def ma_cohens_d(m1, m2, s1, s2, n1, n2):
    """Cohen's d and Hedges' g for two independent groups.

    Parameters
    ----------
    m1, m2 : float
        Group means.
    s1, s2 : float
        Group standard deviations (denominator n-1).
    n1, n2 : int
        Group sizes.  Both must be at least 2, and n1 + n2 at least 3,
        otherwise the pooled variance has no degrees of freedom.

    Returns
    -------
    RichResult
        Keys ``d``, ``s_pooled``, ``var_d``, ``se_d``, ``j``,
        ``j_approx``, ``hedges_g``, ``var_g``, ``se_g``, ``df``, ``n``,
        ``method``.
    """
    n1 = int(n1)
    n2 = int(n2)
    if n1 < 2 or n2 < 2:
        raise ValueError("each group needs at least 2 observations")
    df = n1 + n2 - 2
    m1 = float(m1)
    m2 = float(m2)
    s1 = float(s1)
    s2 = float(s2)
    if s1 < 0.0 or s2 < 0.0:
        raise ValueError("standard deviations must be non-negative")
    sp2 = ((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / df
    sp = math.sqrt(sp2)
    if sp == 0.0:
        raise ValueError("pooled standard deviation is zero")
    d = (m1 - m2) / sp
    ntot = n1 + n2
    base = 1.0 / n1 + 1.0 / n2
    var_d = base + d * d / (2.0 * ntot)
    # exact Hedges (1981) correction; lgamma keeps it finite for large df
    j = math.exp(math.lgamma(df / 2.0)
                 - 0.5 * math.log(df / 2.0)
                 - math.lgamma((df - 1.0) / 2.0))
    j_approx = 1.0 - 3.0 / (4.0 * df - 1.0)
    g = j * d
    var_g = base + g * g / (2.0 * ntot)
    return RichResult(
        payload={
            "d": d,
            "s_pooled": sp,
            "var_d": var_d,
            "se_d": math.sqrt(var_d),
            "j": j,
            "j_approx": j_approx,
            "hedges_g": g,
            "var_g": var_g,
            "se_g": math.sqrt(var_g),
            "df": df,
            "n": ntot,
            "method": "Cohen's d / Hedges' g, two independent groups",
        }
    )


def cheatsheet():
    return "macohd: Cohen's d standardised mean difference"


# compact alias per ledger/NAMING.md
macohensd = ma_cohens_d
