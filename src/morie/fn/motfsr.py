r"""Motif discovery by fitting a two-component mixture model with EM (MM/MEME).

Bailey, T. L., & Elkan, C. (1994) "Fitting a mixture model by expectation
maximization to discover motifs in biopolymers", *Proceedings of the Second
International Conference on Intelligent Systems for Molecular Biology*
(ISMB-94), AAAI Press, 28-36.

Bailey, T. L., & Elkan, C. (1995) "The value of prior knowledge in
discovering motifs with MEME", *Proceedings of the Third International
Conference on Intelligent Systems for Molecular Biology* (ISMB-95), AAAI
Press, 21-29. (Used here only for the search grid over
:math:`\lambda^{(0)}`, which the 1994 paper gives as a range and the 1995
paper as an explicit doubling loop.)

The problem is unsupervised: given unaligned DNA or protein sequences, find
a shared subsequence pattern of fixed width :math:`W` without being told
where -- or even whether -- it occurs in any given sequence. The 1994 paper
solves it by *not* modelling the sequences at all. The dataset
:math:`Y = (Y_1, \dots, Y_N)` is broken conceptually into the :math:`n`
overlapping subsequences of length :math:`W` it contains,
:math:`X = (X_1, \dots, X_n)`, and a **two-component finite mixture** is fitted
to :math:`X`: with probability :math:`\lambda_1` a sample comes from the motif,
with probability :math:`\lambda_2 = 1 - \lambda_1` from the background.

That is the move that distinguishes MM from Lawrence & Reilly (1990): it
"relaxes the assumption that each sequence in the dataset contains one
occurrence of the motif", so sequences with zero, one or many occurrences are
all modelled equally well, and :math:`\lambda_1` -- an estimated quantity, not
an input -- says how often the motif occurs.

The two components (equations 7 and 8) are products of independent
multinomials, position-specific for the motif and position-independent for the
background:

.. math::

   p(X_i \mid \theta_1) = \prod_{j=1}^{W} \prod_{k=1}^{L}
       f_{jk}^{\,I(k, X_{ij})}, \qquad
   p(X_i \mid \theta_2) = \prod_{j=1}^{W} \prod_{k=1}^{L}
       f_{0k}^{\,I(k, X_{ij})},

with :math:`I(k, a) = 1` iff :math:`a = a_k`. EM then alternates

* **E-step** (equation 4), the posterior that :math:`X_i` came from
  component :math:`j`:

  .. math:: Z_{ij}^{(0)} = \frac{p(X_i \mid \theta_j^{(0)})\lambda_j^{(0)}}
            {\sum_{k=1}^{2} p(X_i \mid \theta_k^{(0)})\lambda_k^{(0)}};

* **M-step**, the mixing parameter (equation 5) and the letter frequencies
  (equations 9-13):

  .. math:: \lambda_j^{(1)} = \frac{\sum_{i=1}^{n} Z_{ij}^{(0)}}{n},
            \qquad
            \hat f_{jk} = \frac{c_{jk} + \beta_k}{\sum_{k=1}^{L} c_{jk} + \beta},

  where :math:`c_{0k} = \sum_i \sum_j Z_{i2}^{(0)} I(k, X_{ij})` counts
  background letters and
  :math:`c_{jk} = \sum_i \varepsilon_i Z_{i1}^{(0)} I(k, X_{ij})` counts motif
  letters at position :math:`j`.

Three details in the paper are easy to drop and are implemented here, because
each one is load-bearing:

**Pseudo-counts are not cosmetic.** "If any letter frequency :math:`f_{ij}`
ever becomes 0, as is prone to happen in small datasets, its value can never
change" -- a zero is an absorbing state of the M-step. Equation 13 adds
:math:`\beta_k = \beta \mu_k`, with :math:`\mu_k` the average frequency of
letter :math:`a_k` in the dataset; this is the Bayes estimate under
squared-error loss with a Dirichlet prior.

**Overlapping subsequences are not independent.** The :math:`z` values are
normalised so that :math:`\sum_{j=k}^{k+W-1} z_{ij} \le 1` for every window,
"because otherwise there is a strong tendency for MM to converge to motif
models that generate repeated strings of one or two letters like 'AAAAAA' or
'ATATAT'". Available as ``normalize_overlaps``, on by default; turning it off
gives the plain mixture EM, whose likelihood is monotone.

**Erasing gives multiple motifs.** After a pass, the erasing factors are
reduced by the probability that a position lies in an occurrence of the motif
just found,

.. math:: e_{ij}^{(t)} = e_{ij}^{(t-1)}
          \prod_{k=j-W+1}^{j} \left(1 - z_{ik}\right),

so the next pass is fitted to what is left. Background counts are deliberately
*not* scaled by the erasing factors, "to make the values of the log likelihood
function comparable among passes".

The output is a classifier, not just a matrix. The log-odds matrix
:math:`\mathrm{spec}_{ij} = \log(f_{ij} / f_{0j})` scores a subsequence by
:math:`s(x) = \sum_j \mathrm{spec}_{j, x_j}`, which is exactly
:math:`\log(p(x \mid \theta_1) / p(x \mid \theta_2))`, and Bayesian decision
theory under zero-one loss classifies :math:`x` as an occurrence iff
:math:`s(x) \ge t` with :math:`t = \log((1 - \lambda_1)/\lambda_1)`. For a
general loss matrix :math:`r_{ij}` (the loss for deciding class :math:`i` when
the truth is :math:`j`, class 1 the motif) the threshold scales to
:math:`t + \log\frac{r_{12} - r_{22}}{r_{21} - r_{11}}`.

One boundary is stated rather than hidden: the paper's heuristic for scoring a
candidate :math:`\theta^{(0)}` before running EM is not published (the 1994
paper refers to a longer version, and the 1995 paper omits it "due to space
limitations"). ``start_scoring="one_step"`` therefore ranks candidate starting
points by the mixture likelihood after a single EM iteration, which is a
documented choice here and not a claim about what MEME does internally.
"""

import math

from ._richresult import RichResult

__all__ = ["motfsr", "motif_meme", "mm_fit", "log_odds_matrix",
           "bayes_threshold", "score_sequence"]

_NEG_INF = float("-inf")


def _alphabet_of(seqs, alphabet):
    if alphabet is not None:
        alpha = [str(a) for a in alphabet]
        if len(set(alpha)) != len(alpha):
            raise ValueError("motfsr: alphabet has repeated letters")
        return alpha
    seen = set()
    for s in seqs:
        seen.update(s)
    if not seen:
        raise ValueError("motfsr: the sequences contain no letters")
    return sorted(seen)


def _prepare(sequences, w, alphabet):
    seqs = [str(s) for s in sequences]
    if not seqs:
        raise ValueError("motfsr: sequences must be non-empty")
    w = int(w)
    if w < 1:
        raise ValueError("motfsr: w (motif width) must be >= 1")
    alpha = _alphabet_of(seqs, alphabet)
    idx = dict((a, k) for k, a in enumerate(alpha))
    coded = []
    for s in seqs:
        row = []
        for ch in s:
            if ch not in idx:
                raise ValueError("motfsr: letter %r is not in the alphabet %r"
                                 % (ch, "".join(alpha)))
            row.append(idx[ch])
        coded.append(row)
    starts = [(i, j) for i, row in enumerate(coded)
              for j in range(len(row) - w + 1)]
    if not starts:
        raise ValueError("motfsr: no sequence is at least w = %d long" % w)
    return coded, alpha, starts


def _mu(coded, L):
    """Average frequency of each letter in the dataset (for beta_k)."""
    c = [0.0] * L
    for row in coded:
        for k in row:
            c[k] += 1.0
    tot = sum(c)
    return [x / tot for x in c]


def _uniform_theta(w, L, mu):
    return [list(mu) for _ in range(w + 1)]


def _theta_from_subsequence(coded, i, j, w, L, mu, weight):
    """theta^(0) 'derived from subsequences in the dataset' (section 4).

    Row 0 is the background, taken as the dataset letter frequencies; row
    j >= 1 puts ``weight`` on the letter actually seen at position j of the
    chosen subsequence and spreads the rest over the other letters. The
    paper does not print this construction (it defers to a longer version),
    so ``weight`` is exposed rather than buried.
    """
    theta = [list(mu)]
    for t in range(w):
        k = coded[i][j + t]
        rest = (1.0 - weight) / (L - 1) if L > 1 else 0.0
        row = [rest] * L
        row[k] = weight if L > 1 else 1.0
        theta.append(row)
    return theta


def _log_component(theta, coded, i, j, w, comp):
    """log p(X_i | theta_comp) for the subsequence at (i, j)."""
    tot = 0.0
    for t in range(w):
        k = coded[i][j + t]
        f = theta[t + 1][k] if comp == 1 else theta[0][k]
        if f <= 0.0:
            return _NEG_INF
        tot += math.log(f)
    return tot


def _normalise_windows(z, w, max_sweeps=100):
    """Enforce sum_{j=k}^{k+W-1} z_ij <= 1 for every window (section 4).

    The paper states the constraint and attributes it to Bailey & Elkan
    (1993) without printing the procedure. Repeatedly rescaling the most
    violated window is used here: every sweep strictly reduces the largest
    window sum, and the loop stops when no window exceeds 1.
    """
    if w < 2:
        return z
    for _ in range(max_sweeps):
        worst = 1.0 + 1e-12
        wi = wj = -1
        for i, row in enumerate(z):
            m = len(row)
            if m < w:
                run = sum(row)
                if run > worst:
                    worst, wi, wj = run, i, 0
                continue
            run = sum(row[:w])
            if run > worst:
                worst, wi, wj = run, i, 0
            for j in range(1, m - w + 1):
                run += row[j + w - 1] - row[j - 1]
                if run > worst:
                    worst, wi, wj = run, i, j
        if wi < 0:
            break
        hi = min(wj + w, len(z[wi]))
        for j in range(wj, hi):
            z[wi][j] /= worst
    return z


def mm_fit(sequences, w, alphabet=None, theta0=None, lambda0=None,
           beta=0.01, erasing=None, max_iter=1000, tol=1e-6,
           normalize_overlaps=True, erase_by="letter"):
    r"""One pass of MM: fit the two-component mixture by EM.

    Parameters
    ----------
    sequences : sequence of str
        The dataset :math:`Y`. Sequences may differ in length.
    w : int
        Motif width :math:`W`.
    alphabet : sequence of str, optional
        :math:`A = (a_1, \dots, a_L)`. Defaults to the sorted set of letters
        that occur, which is right for a dataset that uses its whole
        alphabet and wrong for a small one -- pass it explicitly for DNA or
        protein.
    theta0, lambda0 : optional
        Starting point. ``theta0`` is a list of :math:`W + 1` frequency
        vectors, row 0 the background; ``lambda0`` is :math:`\lambda_1^{(0)}`.
    beta : float
        The pseudo-count total of equation 13; :math:`\beta_k = \beta\mu_k`
        with :math:`\mu_k` the average frequency of :math:`a_k` in the
        dataset. Zero reproduces equation 12 and reinstates the boundary
        problem the paper warns about.
    erasing : list of list of float, optional
        The erasing factors :math:`\varepsilon`, one per position, all 1
        when absent.
    erase_by : {"letter", "start"}
        How :math:`\varepsilon` enters the motif counts of equation 10,
        :math:`c_{jk} = \sum_i \varepsilon_i z_{i1} I(k, X_{ij})`. The
        subscript is ambiguous in the paper: :math:`\varepsilon_i` is
        indexed by the *sample* :math:`X_i`, but the footnote calls it "the
        erasing factor for that position in the data", and the erasing
        factors are stored "re-subscripted analogously to :math:`z_{ik}`",
        i.e. one per position of the original sequence. ``"start"`` is the
        literal reading, one factor per sample, taken at the sample's start
        position. ``"letter"`` takes each counted letter's own erasing
        factor, so a subsequence *shifted by one* off an erased occurrence
        is erased too; that is the default because under ``"start"`` a
        shifted copy of an already-discovered motif survives into the next
        pass, since the position one before an occurrence is not covered by
        it.
    max_iter, tol : int, float
        Stop when the Euclidean change in :math:`\theta` falls below ``tol``
        or ``max_iter`` iterations are reached (the paper's defaults, 1000
        and 1e-6).
    normalize_overlaps : bool
        Apply the window constraint of section 4.

    Returns
    -------
    dict
        ``theta``, ``lambda1``, ``z`` (per sequence, per start),
        ``log_likelihood``, ``log_likelihood_trace``, ``n_iter``,
        ``converged``.
    """
    coded, alpha, starts = _prepare(sequences, w, alphabet)
    L = len(alpha)
    w = int(w)
    beta = float(beta)
    if beta < 0.0:
        raise ValueError("motfsr: beta must be >= 0")
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("motfsr: tol must be > 0")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("motfsr: max_iter must be >= 1")
    n = len(starts)
    mu = _mu(coded, L)

    if theta0 is None:
        theta = _theta_from_subsequence(coded, starts[0][0], starts[0][1],
                                        w, L, mu, 0.5)
    else:
        theta = [list(map(float, row)) for row in theta0]
        if len(theta) != w + 1 or any(len(r) != L for r in theta):
            raise ValueError("motfsr: theta0 must be (w + 1) x L")
    lam1 = float(lambda0) if lambda0 is not None else 1.0 / (2.0 * w)
    if not 0.0 < lam1 < 1.0:
        raise ValueError("motfsr: lambda0 must lie in (0, 1)")

    if erase_by not in ("letter", "start"):
        raise ValueError("motfsr: erase_by must be 'letter' or 'start'")
    if erasing is None:
        eps = None
    else:
        eps = [[float(v) for v in row] for row in erasing]

    trace = []
    converged = False
    it = 0
    z_by_seq = None
    for it in range(1, max_iter + 1):
        # ---- E-step (equation 4), in logs -------------------------------
        z_by_seq = [[0.0] * max(0, len(row) - w + 1) for row in coded]
        loglik = 0.0
        log_l1 = math.log(lam1)
        log_l2 = math.log(1.0 - lam1)
        for (i, j) in starts:
            a = log_l1 + _log_component(theta, coded, i, j, w, 1)
            b = log_l2 + _log_component(theta, coded, i, j, w, 2)
            m = a if a > b else b
            if m == _NEG_INF:
                z_by_seq[i][j] = 0.0
                continue
            ea = math.exp(a - m)
            eb = math.exp(b - m)
            z_by_seq[i][j] = ea / (ea + eb)
            loglik += m + math.log(ea + eb)
        trace.append(loglik)
        if normalize_overlaps:
            z_by_seq = _normalise_windows(z_by_seq, w)

        # ---- M-step -----------------------------------------------------
        # equation 5
        z_sum = sum(z_by_seq[i][j] for (i, j) in starts)
        lam1 = z_sum / n
        lam1 = min(max(lam1, 1e-12), 1.0 - 1e-12)
        # equations 9 and 10; erasing scales the MOTIF counts only
        c = [[0.0] * L for _ in range(w + 1)]
        for (i, j) in starts:
            z1 = z_by_seq[i][j]
            z2 = 1.0 - z1
            e = 1.0 if eps is None else (
                eps[i][j] if erase_by == "start" else None)
            for t in range(w):
                k = coded[i][j + t]
                if eps is not None and erase_by == "letter":
                    e = eps[i][j + t]
                c[t + 1][k] += e * z1
                c[0][k] += z2
        # equation 13
        new = []
        for row in c:
            denom = sum(row) + beta
            if denom <= 0.0:
                new.append(list(mu))
                continue
            new.append([(row[k] + beta * mu[k]) / denom for k in range(L)])
        delta = math.sqrt(sum((new[r][k] - theta[r][k]) ** 2
                              for r in range(w + 1) for k in range(L)))
        theta = new
        if delta < tol:
            converged = True
            break

    return {
        "theta": theta,
        "motif": [list(r) for r in theta[1:]],
        "background": list(theta[0]),
        "lambda1": lam1,
        "z": z_by_seq,
        "log_likelihood": trace[-1] if trace else float("nan"),
        "log_likelihood_trace": trace,
        "n_iter": it,
        "converged": converged,
        "alphabet": alpha,
        "w": w,
    }


def log_odds_matrix(motif, background):
    r"""The classifier matrix :math:`\mathrm{spec}_{ij} = \log(f_{ij}/f_{0j})`.

    Rows are motif positions, columns letters.
    """
    out = []
    for row in motif:
        r = []
        for k, f in enumerate(row):
            b = background[k]
            if f <= 0.0:
                r.append(_NEG_INF)
            elif b <= 0.0:
                r.append(float("inf"))
            else:
                r.append(math.log(f / b))
        out.append(r)
    return out


def bayes_threshold(lambda1, loss=None):
    r"""Bayes-optimal threshold :math:`t = \log((1 - \lambda_1)/\lambda_1)`.

    ``loss`` is the matrix :math:`r` with :math:`r_{ij}` the loss for
    deciding class :math:`i` when the truth is :math:`j` and class 1 the
    motif, given as ``[[r11, r12], [r21, r22]]``; the threshold then becomes
    :math:`t + \log\frac{r_{12} - r_{22}}{r_{21} - r_{11}}`. Omit it for
    zero-one loss.
    """
    lambda1 = float(lambda1)
    if not 0.0 < lambda1 < 1.0:
        raise ValueError("motfsr: lambda1 must lie in (0, 1)")
    t = math.log((1.0 - lambda1) / lambda1)
    if loss is None:
        return t
    (r11, r12), (r21, r22) = loss
    num = float(r12) - float(r22)
    den = float(r21) - float(r11)
    if num <= 0.0 or den <= 0.0:
        raise ValueError("motfsr: the loss matrix must have r12 > r22 and "
                         "r21 > r11 for the threshold to be defined")
    return t + math.log(num / den)


def score_sequence(spec, sequence, alphabet, threshold=None):
    r"""Score every subsequence of ``sequence`` with ``spec``.

    :math:`s(x) = \sum_{j} \mathrm{spec}_{j, x_j}`, which equals
    :math:`\log(p(x \mid \theta_1) / p(x \mid \theta_2))`. With a threshold,
    positions scoring at least ``threshold`` are returned as hits.
    """
    idx = dict((a, k) for k, a in enumerate(alphabet))
    w = len(spec)
    s = str(sequence)
    scores = []
    for j in range(len(s) - w + 1):
        tot = 0.0
        for t in range(w):
            ch = s[j + t]
            if ch not in idx:
                raise ValueError("motfsr: letter %r is not in the alphabet"
                                 % (ch,))
            tot += spec[t][idx[ch]]
        scores.append(tot)
    if threshold is None:
        return scores
    return scores, [j for j, v in enumerate(scores) if v >= threshold]


def _lambda_grid(n_starts_total, n_seqs, w, lambda0):
    r"""The :math:`\lambda^{(0)}` search grid.

    The 1994 paper (section 4) searches values "between :math:`\sqrt{N}/n`
    and :math:`1/(2W)`"; the 1995 paper's procedure walks such a range by
    doubling. Both endpoints are kept, and doubling is the step.
    """
    if lambda0 is not None:
        return [float(lambda0)]
    lo = math.sqrt(n_seqs) / n_starts_total
    hi = 1.0 / (2.0 * w)
    lo = min(max(lo, 1e-9), 0.5)
    hi = min(max(hi, lo), 0.5)
    out = []
    v = lo
    while v < hi:
        out.append(v)
        v *= 2.0
    out.append(hi)
    return out


def motfsr(sequences, w, alphabet=None, n_motifs=1, beta=0.01, lambda0=None,
           max_iter=1000, tol=1e-6, normalize_overlaps=True,
           starts="subsequences", start_weight=0.5, max_starts=200,
           start_scoring="one_step", erase_by="letter", loss=None):
    r"""Discover motifs by fitting the MM mixture model (Bailey & Elkan 1994).

    Runs one MM pass per motif, and between passes erases the occurrences of
    the motif just found -- the MEME+ loop of section 4 -- so successive
    motifs may occur different numbers of times.

    Parameters
    ----------
    sequences : sequence of str
        The unaligned dataset.
    w : int
        Motif width. The one number the algorithm requires beyond the data.
    alphabet : sequence of str, optional
        Pass it explicitly (``"ACGT"``, the 20 amino acids) unless the
        dataset is guaranteed to use every letter.
    n_motifs : int
        Number of passes; each erases what it found before the next.
    beta : float
        Pseudo-count total of equation 13. See :func:`mm_fit`.
    lambda0 : float, optional
        Fix :math:`\lambda^{(0)}` instead of searching the grid.
    max_iter, tol : int, float
        EM stopping rule.
    normalize_overlaps : bool
        The window constraint of section 4.
    starts : {"subsequences", "uniform"}
        Where :math:`\theta^{(0)}` comes from: derived from subsequences of
        the dataset, as in the paper, or a single uniform (background) start,
        which is the fastest thing that can work and is much more prone to a
        poor local optimum.
    start_weight : float
        The probability placed on the observed letter when building
        :math:`\theta^{(0)}` from a subsequence.
    max_starts : int
        Cap on how many subsequences are tried, evenly spaced through the
        dataset. The full search is :math:`O(n)` EM iterations per
        :math:`\lambda^{(0)}`.
    erase_by : {"letter", "start"}
        How the erasing factors enter the motif counts; see :func:`mm_fit`.
    start_scoring : {"one_step", "none"}
        How candidates are ranked before the full run. ``"one_step"`` runs a
        single EM iteration and takes the best mixture likelihood;
        ``"none"`` runs every candidate to convergence and takes the best,
        which is exact and slow. The paper's own heuristic is unpublished.
    loss : list of list of float, optional
        Loss matrix for the classifier threshold; see
        :func:`bayes_threshold`.

    Returns
    -------
    RichResult
        ``estimate`` / ``motifs`` is a list, one per pass, each with
        ``motif`` (the :math:`W \times L` frequency matrix), ``background``,
        ``lambda1``, ``log_odds``, ``threshold``, ``sites`` (the
        ``(sequence, position, score)`` triples the Bayes classifier accepts),
        ``consensus``, ``log_likelihood`` and ``n_iter``. ``alphabet`` and
        ``w`` describe the fit.

    Examples
    --------
    A planted motif is recovered from sequences that also contain background
    only::

        seqs = ["TTTTACGTGTTTT", "AAACGTGAAAAAA", "GGGGGGACGTGCC"]
        r = motfsr(seqs, w=5, alphabet="ACGT")
        r["motifs"][0]["consensus"]           # "ACGTG"

    The fitted model is a classifier for other data::

        m = r["motifs"][0]
        scores, hits = score_sequence(m["log_odds"], "CCACGTGCC",
                                      r["alphabet"], m["threshold"])

    References
    ----------
    Bailey & Elkan (1994) ISMB-94, 28-36, equations 4, 5, 7-13 and section 4;
    Bailey & Elkan (1995) ISMB-95, 21-29, for the :math:`\lambda^{(0)}` grid.
    """
    if starts not in ("subsequences", "uniform"):
        raise ValueError("motfsr: starts must be 'subsequences' or 'uniform'")
    if start_scoring not in ("one_step", "none"):
        raise ValueError("motfsr: start_scoring must be 'one_step' or 'none'")
    n_motifs = int(n_motifs)
    if n_motifs < 1:
        raise ValueError("motfsr: n_motifs must be >= 1")
    start_weight = float(start_weight)
    if not 0.0 < start_weight < 1.0:
        raise ValueError("motfsr: start_weight must lie in (0, 1)")

    coded, alpha, all_starts = _prepare(sequences, w, alphabet)
    L = len(alpha)
    w = int(w)
    mu = _mu(coded, L)
    n = len(all_starts)
    # one erasing factor per LETTER position, which is what "the erasing
    # factor for that position in the data" means and what the product
    # over k in [j - W + 1, j] computes: the probability that position j
    # is not inside an occurrence of a motif found so far.
    erasing = [[1.0] * len(row) for row in coded]

    if starts == "uniform":
        cand = [_uniform_theta(w, L, mu)]
    else:
        step = max(1, int(math.ceil(n / float(max(1, int(max_starts))))))
        cand = [_theta_from_subsequence(coded, i, j, w, L, mu, start_weight)
                for (i, j) in all_starts[::step]]
    lam_grid = _lambda_grid(n, len(coded), w, lambda0)

    motifs = []
    for _pass in range(n_motifs):
        best = None
        for th0 in cand:
            for lam in lam_grid:
                if start_scoring == "one_step":
                    probe = mm_fit(sequences, w, alpha, th0, lam, beta,
                                   erasing, 1, tol, normalize_overlaps,
                                   erase_by)
                else:
                    probe = mm_fit(sequences, w, alpha, th0, lam, beta,
                                   erasing, max_iter, tol,
                                   normalize_overlaps, erase_by)
                key = probe["log_likelihood"]
                if best is None or key > best[0]:
                    best = (key, th0, lam, probe)
        _, th0, lam, probe = best
        fit = (probe if start_scoring == "none" else
               mm_fit(sequences, w, alpha, th0, lam, beta, erasing,
                      max_iter, tol, normalize_overlaps, erase_by))

        spec = log_odds_matrix(fit["motif"], fit["background"])
        t = bayes_threshold(fit["lambda1"], loss)
        sites = []
        for i, row in enumerate(fit["z"]):
            for j in range(len(row)):
                s = 0.0
                for q in range(w):
                    s += spec[q][coded[i][j + q]]
                if s >= t:
                    sites.append((i, j, s))
        sites.sort(key=lambda r: -r[2])
        consensus = "".join(
            alpha[max(range(L), key=lambda k: row[k])] for row in fit["motif"])
        motifs.append({
            "motif": fit["motif"],
            "background": fit["background"],
            "lambda1": fit["lambda1"],
            "log_odds": spec,
            "threshold": t,
            "sites": sites,
            "n_sites": len(sites),
            "consensus": consensus,
            "log_likelihood": fit["log_likelihood"],
            "n_iter": fit["n_iter"],
            "converged": fit["converged"],
            "z": fit["z"],
        })

        if _pass + 1 < n_motifs:
            # erasing (section 4): e_ij^(t) = e_ij^(t-1) prod_{k=j-W+1}^{j}
            # (1 - z_ik)
            z = fit["z"]
            for i, row in enumerate(erasing):
                for j in range(len(row)):
                    f = 1.0
                    for k in range(max(0, j - w + 1), min(j + 1, len(z[i]))):
                        f *= (1.0 - z[i][k])
                    erasing[i][j] = row[j] * f

    return RichResult(payload={
        "estimate": motifs,
        "motifs": motifs,
        "alphabet": alpha,
        "w": w,
        "n_subsequences": n,
        "erasing": erasing,
        "method": "MM two-component mixture EM (Bailey & Elkan 1994)",
    })


def cheatsheet():
    return ("motfsr: MEME/MM motif discovery (Bailey & Elkan 1994). Break "
            "the sequences into ALL overlapping W-mers and fit a "
            "two-component mixture -- motif vs background -- by EM, so a "
            "sequence may contain zero, one or many occurrences and "
            "lambda1 estimates how often. E-step eq.4; M-step eq.5 for "
            "lambda and eq.13 for the letter frequencies, whose "
            "pseudo-counts exist because a frequency that hits 0 can never "
            "leave. z is normalised to sum to <= 1 over any W-window or EM "
            "collapses onto 'AAAAAA'. Multiple motifs come from erasing. "
            "Output is a Bayes-optimal classifier: log-odds matrix plus "
            "t = log((1 - lambda1)/lambda1).")


# compact alias per ledger/NAMING.md
motif_meme = motfsr
