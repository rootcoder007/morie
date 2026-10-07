Population Genetics
===================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE provides population genetics functions for analyzing genetic variation,
population structure, and genotype-phenotype associations. All functions are
dataset-agnostic with column names as keyword parameters.

Sequence-Level Metrics
----------------------

- ``gc`` — GC content. Proportion of guanine + cytosine in a DNA sequence.
- ``maf`` — minor allele frequency. Frequency of the less common allele at a locus.
- ``hw`` — Hardy-Weinberg equilibrium. Chi-squared test for HWE departure (observed vs expected genotype counts).

.. code-block:: python

   from morie.fn import gc_content_calc, maf_calculation, hardy_weinberg_test
   from morie.fn import _array_core as np

   gc = gc_content_calc("ATGCGCTATGCGC")
   print(f"GC content: {gc.estimate:.3f}")   # 0.615

   markers = np.array([[0, 1, 2], [1, 1, 0], [2, 0, 1], [0, 1, 1]])   # individuals x SNPs, 0/1/2 copies
   print(maf_calculation(markers))

   hwe = hardy_weinberg_test(45, 40, 15)   # AA, Aa, aa counts
   print(f"HWE chi2={hwe.statistic:.2f}, p={hwe.p_value:.4f}")

Population Differentiation
--------------------------

- ``fst`` — fixation index. Weir-Cockerham :math:`F_{ST}` estimator for population differentiation.
- ``tajd`` — Tajima's D. Neutrality test comparing pairwise diversity to segregating sites.

.. code-block:: python

   from morie.fn import tajimas_d

   taj = tajimas_d(S=42, n=50, pi=18.5)   # segregating sites, sequences, mean pairwise differences
   print(f"Tajima's D = {taj.statistic:.3f}")

Linkage Disequilibrium
----------------------

- ``ld`` — linkage disequilibrium. D, D', and r-squared between two loci.
- ``ldmat`` — LD matrix. Pairwise r-squared matrix for a set of SNPs.

.. code-block:: python

   from morie.fn import linkage_disequilibrium

   # haplotype alleles (0/1) at two loci, one entry per chromosome
   result = linkage_disequilibrium([0, 0, 1, 1, 1, 0, 1, 1], [0, 1, 1, 1, 1, 0, 1, 0])
   print(f"r2 = {result.statistic:.3f}")

Genome-Wide Association Studies
-------------------------------

- ``gwas`` — GWAS scan. Per-SNP association test (linear or logistic) with multiple-testing correction.
- ``prs`` — polygenic risk score. Weighted sum of risk alleles using GWAS summary statistics.

.. code-block:: python

   from morie.fn import gwas_single_snp
   from morie.fn import _array_core as np

   genotypes = np.array([0, 1, 2, 1, 0, 2, 1, 1, 0, 2])
   phenotype = np.array([1.2, 1.9, 3.1, 2.2, 0.8, 2.9, 1.7, 2.3, 1.1, 3.0])
   hit = gwas_single_snp(genotypes, phenotype)   # additive linear model, one SNP
   print(f"beta = {hit.extra['beta']:.3f}, p = {hit.p_value:.2e}")
   # prs_cs(beta_hat, D, psi, n) gives continuous-shrinkage polygenic effects from summary statistics

**GWAS pipeline:**

1. Quality control: MAF filter, HWE filter, missingness filter
2. Association testing: per-SNP regression (``gwas``)
3. Multiple testing correction: Bonferroni or Benjamini-Hochberg
4. Visualization: Manhattan plot, QQ plot (via ``holo_*`` functions)
5. Risk prediction: polygenic risk scores (``prs``)

Epidemiological Applications
----------------------------

Population genetics functions integrate with MORIE's epidemiological
toolkit for public health genomics:

- **Pharmacogenomics**: MAF and HWE checks for drug-metabolizing enzyme
  variants across population subgroups
- **Disease surveillance**: Fst for tracking pathogen population structure
  across geographic regions
- **Health equity**: PRS calibration across ancestry groups to avoid
  differential prediction accuracy

All functions return standardized result objects (``GenomicsResult``
dataclass from ``_containers.py``) with ``statistic``, ``p_value``,
and method-specific fields.

References
----------

.. [Weir1984] Weir, B.S. & Cockerham, C.C. (1984). Estimating F-Statistics
   for the Analysis of Population Structure. *Evolution*, 38(6), 1358-1370.

.. [Tajima1989] Tajima, F. (1989). Statistical Method for Testing the Neutral
   Mutation Hypothesis by DNA Polymorphism. *Genetics*, 123(3), 585-595.

.. [Purcell2007] Purcell, S. et al. (2007). PLINK: A Tool Set for
   Whole-Genome Association and Population-Based Linkage Analyses.
   *American Journal of Human Genetics*, 81(3), 559-575.
