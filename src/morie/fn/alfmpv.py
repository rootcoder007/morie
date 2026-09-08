"""AlphaFold-Multimer chain pairing, and the two published alternatives.

A complex is predicted from one MSA covering every chain. Rows that are
PAIRED -- one sequence per chain, all from the same organism -- carry the
inter-chain co-evolutionary signal; everything else is stacked block
diagonally, present for the intra-chain signal and gapped elsewhere. How
rows get paired is the whole method, and three papers specify it three
different ways, so all three are here and the caller picks.

  multimer  Evans et al. (2022). Group hits by their UniProt OX, rank
            within a species by similarity to that chain's query, and
            join same-rank rows across chains.

  colabfold Mirdita et al. (2022). Keep only species that cover every
            chain, require the alignment to cover at least half the
            query, and pair ONLY the best hit -- smallest E-value -- so a
            species contributes exactly one row.

  folddock  Bryant et al. (2022). Drop hits that are more than 90% gaps,
            then pair the highest-ranked hit from one organism with the
            highest-ranked hit of the interacting chain from the same
            organism. Again one row per organism.

The edge case worth knowing about: AlphaFold-Multimer's rule is stated for
same-rank rows but the paper does not say what to do when a species has a
different number of hits for each chain. Neither does ColabFold or
FoldDock -- they sidestep it by construction, since taking one hit per
species means the counts cannot disagree. Only the multimer route has to
decide, and it pairs up to the smallest count in that species and leaves
the surplus for the block-diagonal region. That is this implementation's
reading, not something the paper states, and it is reported as
pairing_rule so nobody mistakes it for a quotation.

Indices in the result are 0-based, which is this package's reported-field
convention regardless of the language the arm is written in.

References
  Evans, R. et al. (2022) "Protein complex prediction with
    AlphaFold-Multimer." bioRxiv 2021.10.04.463034v2, section 2.1.
    doi:10.1101/2021.10.04.463034
  Mirdita, M., Schutze, K., Moriwaki, Y., Heo, L., Ovchinnikov, S. &
    Steinegger, M. (2022) "ColabFold: making protein folding accessible
    to all." Nature Methods 19(6), 679-682.
    doi:10.1038/s41592-022-01488-1
  Bryant, P., Pozzati, G. & Elofsson, A. (2022) "Improved prediction of
    protein-protein interactions using AlphaFold2." Nature Communications
    13, 1265. doi:10.1038/s41467-022-28865-w
"""

from ._richresult import RichResult

__all__ = ["alphafold_multimer", "alfmpv", "msa_pairing", "cheatsheet"]

_MODES = ("multimer", "colabfold", "folddock")


def _field(chain, name, n, default):
    """One hit attribute as a list of length n, defaulted when absent.

    A caller who has no E-values should still be able to run the multimer
    route, which never looks at them.
    """
    v = chain.get(name) if hasattr(chain, "get") else None
    if v is None:
        return [default] * n
    v = list(v)
    if len(v) != n:
        raise ValueError(
            "alfmpv: chain field %s has %d entries but %d hits"
            % (name, len(v), n))
    return v


def _chain_table(chain, idx):
    if hasattr(chain, "get") and chain.get("species") is not None:
        species = [str(s) for s in chain["species"]]
    elif isinstance(chain, (list, tuple)):
        species = [str(s) for s in chain]
        chain = {}
    else:
        raise ValueError("alfmpv: chain %d has no species field" % idx)
    n = len(species)
    return {
        "species": species,
        "evalue": [float(v) for v in _field(chain, "evalue", n, 0.0)],
        "identity": [float(v) for v in _field(chain, "identity", n, 0.0)],
        "gaps": [float(v) for v in _field(chain, "gaps", n, 0.0)],
        "coverage": [float(v) for v in _field(chain, "coverage", n, 1.0)],
        "n": n,
    }


def _keep(tab, mode, min_coverage, max_gap):
    """Which hits survive the route's own filter, in order."""
    out = []
    for i in range(tab["n"]):
        if mode == "colabfold" and tab["coverage"][i] < min_coverage:
            continue
        if mode == "folddock" and tab["gaps"][i] > max_gap:
            continue
        out.append(i)
    return out


def _rank_key(tab, mode, i, order):
    """Lower sorts first. Ties break on the hit's position in the MSA, so
    the order is total and neither arm can wander off on a tie."""
    if mode == "multimer":
        return (-tab["identity"][i], order)
    if mode == "colabfold":
        return (tab["evalue"][i], order)
    return (order,)                       # folddock: MSA rank as given


def msa_pairing(msas, mode="multimer", min_coverage=0.5, max_gap=0.9,
                copies=None, max_pairs=None):
    """Pair MSA rows across the chains of a complex.

    Parameters
    ----------
    msas : sequence
        One entry per chain. Each is either a plain sequence of species
        identifiers, or a mapping with a species key and any of evalue,
        identity, gaps and coverage as parallel sequences.
    mode : str
        multimer, colabfold or folddock.
    min_coverage : float
        colabfold only: least fraction of the query the alignment must
        cover. 0.5 is the published value.
    max_gap : float
        folddock only: most gaps a hit may be, as a fraction. 0.9 is the
        published value.
    copies : sequence of int, optional
        Copy count per chain, for a homo-oligomer. ColabFold copies the
        MSA per component rather than searching again; the copies share
        one chain's hits, so a paired row repeats that chain's index.
    max_pairs : int, optional
        Stop after this many paired rows.

    Returns
    -------
    RichResult
        paired : list of rows, each a list of 0-based hit indices, one
            per chain (after copy expansion).
        species_paired : the organism each paired row came from.
        unpaired : per chain, the 0-based indices left for the block
            diagonal region.
        n_paired, n_unpaired, n_rows, n_chains, pairing_rule, mode.
    """
    if mode not in _MODES:
        raise ValueError("alfmpv: mode = %r; expected one of %s"
                         % (mode, ", ".join(_MODES)))
    if not msas:
        raise ValueError("alfmpv: no chains given")

    tabs = [_chain_table(c, i) for i, c in enumerate(msas)]

    # A homo-oligomer reuses one chain's alignment for each copy rather
    # than searching again, so the copies are the same table.
    if copies is not None:
        if len(copies) != len(tabs):
            raise ValueError("alfmpv: copies has %d entries for %d chains"
                             % (len(copies), len(tabs)))
        expanded = []
        source = []
        for j, k in enumerate(copies):
            k = int(k)
            if k < 1:
                raise ValueError("alfmpv: copies[%d] = %d; need at least 1"
                                 % (j, k))
            for _ in range(k):
                expanded.append(tabs[j])
                source.append(j)
        tabs = expanded
    else:
        source = list(range(len(tabs)))

    nc = len(tabs)
    kept = [_keep(t, mode, min_coverage, max_gap) for t in tabs]

    # Species order follows the first chain, so the paired block is in a
    # defined order without sorting strings -- which would put the two
    # arms at the mercy of their collation locales.
    order = []
    seen = {}
    for c in range(nc):
        for pos, i in enumerate(kept[c]):
            s = tabs[c]["species"][i]
            if s not in seen:
                seen[s] = len(order)
                order.append(s)

    by_species = []
    for c in range(nc):
        d = {}
        for pos, i in enumerate(kept[c]):
            d.setdefault(tabs[c]["species"][i], []).append((pos, i))
        for s in d:
            d[s] = [i for _, i in
                    sorted(d[s], key=lambda pi: _rank_key(
                        tabs[c], mode, pi[1], pi[0]))]
        by_species.append(d)

    paired = []
    species_paired = []
    used = [set() for _ in range(nc)]
    for s in order:
        lists = [by_species[c].get(s, []) for c in range(nc)]
        if any(len(l) == 0 for l in lists):
            continue                      # species must cover every chain
        depth = 1 if mode in ("colabfold", "folddock") \
            else min(len(l) for l in lists)
        for k in range(depth):
            if max_pairs is not None and len(paired) >= int(max_pairs):
                break
            row = [lists[c][k] for c in range(nc)]
            paired.append(row)
            species_paired.append(s)
            for c in range(nc):
                used[c].add(row[c])
        if max_pairs is not None and len(paired) >= int(max_pairs):
            break

    unpaired = [[i for i in kept[c] if i not in used[c]] for c in range(nc)]

    rule = ("pair up to the smallest per-species hit count; the surplus "
            "goes block diagonal (this implementation's reading -- Evans "
            "et al. state the same-rank rule but not the unequal case)"
            if mode == "multimer" else
            "one hit per species, so counts cannot disagree")

    return RichResult(payload={
        "paired": paired,
        "species_paired": species_paired,
        "unpaired": unpaired,
        "n_paired": len(paired),
        "n_unpaired": [len(u) for u in unpaired],
        "n_rows": len(paired) + sum(len(u) for u in unpaired),
        "n_chains": nc,
        "chain_source": source,
        "n_filtered": [tabs[c]["n"] - len(kept[c]) for c in range(nc)],
        "mode": mode,
        "pairing_rule": rule,
        "method": {
            "multimer": "AlphaFold-Multimer species pairing (Evans et al. "
                        "2022, section 2.1)",
            "colabfold": "ColabFold best-hit-per-species pairing (Mirdita "
                         "et al. 2022), coverage >= %g" % min_coverage,
            "folddock": "FoldDock top-ranked-per-organism pairing (Bryant "
                        "et al. 2022), gaps <= %g" % max_gap}[mode],
    })


def alphafold_multimer(chains=None, msas=None, mode="multimer", **kw):
    """AlphaFold-Multimer chain pairing.

    chains is accepted as an alias for msas so the older call shape keeps
    working; one of the two has to be given.
    """
    if msas is None:
        msas = chains
    if msas is None:
        raise ValueError("alfmpv: give the per-chain MSAs")
    return msa_pairing(msas, mode=mode, **kw)


alfmpv = alphafold_multimer


def cheatsheet():
    return ("alfmpv: MSA pairing for a complex. mode = multimer (Evans "
            "2022, same-rank within species) | colabfold (Mirdita 2022, "
            "best E-value per species, coverage >= 0.5) | folddock "
            "(Bryant 2022, top-ranked per organism, gaps <= 0.9).")
