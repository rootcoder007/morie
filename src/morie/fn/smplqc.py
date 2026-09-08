# morie.fn -- function file (rootcoder007/morie)
"""Sample quality control: call rate, heterozygosity, inbreeding F."""

import math

from ._richresult import RichResult

__all__ = ["sample_qc"]


def sample_qc(G, callrate_min=0.98, het_sd=3.0, small_sample=False):
    """Per-sample genotype QC (call rate, heterozygosity, F).

    Follows the sample-QC stages of Marees et al. (2018): individuals
    are flagged when their genotype call rate is below
    ``callrate_min`` (Marees step 1 removes missingness > 0.02, i.e.
    call rate < 0.98) and when their heterozygosity rate deviates more
    than ``het_sd`` standard deviations from the sample mean (Marees
    step 5: "removing individuals who deviate +/- 3 SD from the
    samples' heterozygosity rate mean").

    Statistics per individual i over its non-missing variants:

    * call rate = n_obs_i / m;
    * observed homozygote count O_i;
    * expected homozygote count under Hardy-Weinberg
      ``E_i = sum_j (1 - 2 p_j (1 - p_j) c_j)`` over the non-missing
      variants j, with allele frequency p_j estimated from the column
      and ``c_j = N_j / (N_j - 1)`` when ``small_sample`` is True
      (Nei's unbiased expected homozygosity; PLINK 1.9 --het notes the
      multiplier is omitted by default and restored by its
      'small-sample' modifier);
    * method-of-moments inbreeding coefficient
      ``F_i = (O_i - E_i) / (n_obs_i - E_i)`` (PLINK 1.9 --het:
      "(observed hom. count - expected count) / (total observations -
      expected count)");
    * heterozygosity rate ``het_i = (n_obs_i - O_i) / n_obs_i``, the
      quantity Marees et al. compute from PLINK .het output as
      (N(NM) - O(HOM)) / N(NM).

    Parameters
    ----------
    G : (n, m) array-like
        Genotype matrix, individuals by variants, coded 0/1/2; any
        other value is treated as missing.
    callrate_min : float
        Call-rate flag threshold (default 0.98, Marees et al. --mind
        0.02).
    het_sd : float
        Heterozygosity flag width in standard deviations (default 3).
    small_sample : bool
        Apply Nei's N/(N-1) multiplier to expected homozygosity.

    Returns
    -------
    RichResult
        Keys ``estimate`` (number of samples passing), ``callrate``,
        ``het_rate``, ``F``, ``obs_hom``, ``exp_hom``, ``n_obs``,
        ``flag_callrate``, ``flag_het``, ``pass_qc``, ``het_mean``,
        ``het_sd``, ``freq``, ``n``, ``m``, ``method``.

    References
    ----------
    Marees, A. T., de Kluiver, H., Stringer, S., et al. (2018). A
    tutorial on conducting genome-wide association studies: Quality
    control and statistical analysis. International Journal of Methods
    in Psychiatric Research 27(2), e1608, Table 1 steps 1 and 5 and
    sec. "QC of genetic data" (local file
    /run/media/rootcoder/WD_BLACK/library/pdf/fetched-wave3/Marees-2018-GWAS-QC-tutorial-IJMPR27-e1608.pdf).
    PLINK 1.9 basic statistics documentation, --het and --missing,
    https://www.cog-genomics.org/plink/1.9/basic_stats (F formula
    quoted above; fetched 2026-08-09).
    Nei, M. (1978). Estimation of average heterozygosity and genetic
    distance from a small number of individuals. Genetics 89(3),
    583-590 (unbiased expected homozygosity multiplier).
    """
    rows = [[float(v) for v in row] for row in G]
    n = len(rows)
    if n == 0:
        raise ValueError("empty genotype matrix")
    m = len(rows[0])
    valid = (0.0, 1.0, 2.0)
    # per-variant allele frequency and observation count
    freq = []
    nobs_col = []
    for j in range(m):
        obs = [rows[i][j] for i in range(n) if rows[i][j] in valid]
        nobs_col.append(len(obs))
        freq.append(sum(obs) / (2.0 * len(obs)) if obs else float("nan"))
    callrate = []
    het_rate = []
    F = []
    obs_hom = []
    exp_hom = []
    n_obs_list = []
    for i in range(n):
        n_obs = 0
        o_hom = 0
        e_hom = 0.0
        for j in range(m):
            g = rows[i][j]
            if g not in valid:
                continue
            n_obs += 1
            if g != 1.0:
                o_hom += 1
            p = freq[j]
            c = 1.0
            if small_sample and nobs_col[j] > 1:
                c = nobs_col[j] / (nobs_col[j] - 1.0)
            e_hom += 1.0 - 2.0 * p * (1.0 - p) * c
        callrate.append(n_obs / m)
        n_obs_list.append(n_obs)
        obs_hom.append(o_hom)
        exp_hom.append(e_hom)
        den = n_obs - e_hom
        F.append((o_hom - e_hom) / den if den != 0 else float("nan"))
        het_rate.append((n_obs - o_hom) / n_obs if n_obs > 0 else float("nan"))
    het_ok = [h for h in het_rate if h == h]
    hmean = sum(het_ok) / len(het_ok) if het_ok else float("nan")
    if len(het_ok) > 1:
        hsd = math.sqrt(sum((h - hmean) ** 2 for h in het_ok) / (len(het_ok) - 1))
    else:
        hsd = float("nan")
    flag_cr = [cr < float(callrate_min) for cr in callrate]
    flag_het = []
    for h in het_rate:
        if h != h or hsd != hsd:
            flag_het.append(False)
        else:
            flag_het.append(abs(h - hmean) > float(het_sd) * hsd)
    pass_qc = [not (a or b) for a, b in zip(flag_cr, flag_het)]
    return RichResult(payload={
        "estimate": float(sum(1 for p in pass_qc if p)),
        "callrate": callrate, "het_rate": het_rate, "F": F,
        "obs_hom": obs_hom, "exp_hom": exp_hom, "n_obs": n_obs_list,
        "flag_callrate": flag_cr, "flag_het": flag_het,
        "pass_qc": pass_qc, "het_mean": float(hmean), "het_sd": float(hsd),
        "freq": freq, "n": int(n), "m": int(m),
        "method": "Sample QC (Marees 2018 steps 1+5; PLINK --het F)",
    })


def cheatsheet():
    return "smplqc: per-sample call rate, het rate, PLINK F; flags callrate<min and |het-mean|>k SD."


# compact alias per ledger/NAMING.md
sampleqc = sample_qc
smplqc = sample_qc
