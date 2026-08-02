# morie.fn -- function file (rootcoder007/morie)
"""Bootstrap standard errors for Aldrich-McKelvey scaling."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["am_bootstrap_se"]


def am_bootstrap_se(survey_data, B=200, seed=0):
    r"""Respondent-resampling bootstrap for A-M stimulus positions.

    Re-runs Aldrich-McKelvey scaling on B resamples of the respondent
    rows and reports :math:`SE(\hat z_k) = sd(\hat z_k^{(b)})`. Since
    each replicate's scale is only identified up to an affine map,
    every replicate is first normalised (mean 0, sd 1) and sign-aligned
    to the full-sample solution -- without that step the bootstrap sd
    measures the arbitrary normalisation, not sampling noise.

    Parameters
    ----------
    survey_data : array-like, shape (n, q)
        Respondent x stimulus placement matrix (NaN allowed).
    B : int, default 200
        Bootstrap replicates.
    seed : int, default 0

    Returns
    -------
    RichResult
        keys: ``se`` (q,), ``zhat`` (full-sample positions,
        normalised), ``replicates`` (B, q), ``B``, ``n``, ``method``.

    References
    ----------
    Aldrich, J. H. & McKelvey, R. D. (1977). A method of scaling with
    applications to the 1968 and 1972 presidential elections. *APSR*,
    71(1), 111-130.

    Armstrong, D. A. et al. (2014). *Analyzing Spatial Models of
    Choice and Judgment*. CRC Press. Ch. 2 (issue scales and A-M
    scaling), p. 13.
    """
    from morie._spatial_voting import aldrich_mckelvey

    Z = np.asarray(survey_data, dtype=float)
    if Z.ndim != 2:
        raise ValueError("survey_data must be 2-D (respondents x stimuli).")
    n, q = Z.shape
    B = int(B)
    if B < 20:
        raise ValueError(f"B must be at least 20, got {B}.")

    def norm_align(z, ref=None):
        z = np.asarray(z, dtype=float).ravel()
        s = z.std()
        if s <= 0:
            return np.zeros_like(z)
        z = (z - z.mean()) / s
        if ref is not None and np.dot(z, ref) < 0:
            z = -z
        return z

    z_full = norm_align(aldrich_mckelvey(Z)["zhat"])
    rng = np.random.default_rng(seed)
    reps = np.empty((B, q))
    for b in range(B):
        idx = rng.integers(0, n, n)
        reps[b] = norm_align(aldrich_mckelvey(Z[idx])["zhat"], ref=z_full)

    return RichResult(
        payload={
            "se": reps.std(axis=0, ddof=1),
            "zhat": z_full,
            "replicates": reps,
            "B": B,
            "n": int(n),
            "method": "A-M bootstrap SEs (respondent resampling, replicates sign-aligned)",
        }
    )


def cheatsheet():
    return "ambtc: resample respondents, rerun A-M, normalise + sign-align, sd per stimulus"
