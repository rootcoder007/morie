# morie.fn -- function file (rootcoder007/morie)
"""l-diversity of a released table, in all three of the paper's senses.

Source opened: Machanavajjhala, A., Kifer, D., Gehrke, J. and
Venkitasubramaniam, M. (2007).  l-diversity: privacy beyond
k-anonymity.  *ACM Transactions on Knowledge Discovery from Data* 1(1),
article 3, doi:10.1145/1217299.1217302 (author draft, page 11).

Principle 2 says a q*-block is l-diverse if it contains at least l
"well-represented" values of the sensitive attribute.  The paper gives
two concrete instantiations of "well-represented":

  Definition 4.1 (Entropy l-diversity)
      - sum_{s in S} p(q*,s) log p(q*,s) >= log(l)   for every q*-block

  Definition 4.2 (Recursive (c, l)-diversity)
      r_1 < c (r_l + r_{l+1} + ... + r_m)            for every q*-block

  where r_1 >= ... >= r_m are the sensitive-value counts of the block in
  descending order, and 1-diversity is always satisfied.

The paper also notes the consequence of Definition 4.1 that every block
then has at least l distinct sensitive values -- distinct l-diversity,
the weakest of the three.  All three are reported here, because they do
not agree: the paper's own Figure 4 is 3-diverse in the distinct sense
and only 2.8-diverse in the entropy sense, which is the printed number
this module is anchored on.

k-anonymity is reported alongside: l-diversity is a strictly stronger
requirement and a block can be large yet homogeneous.
"""

from __future__ import annotations

import math

from . import _s03core as core
from ._richresult import RichResult

__all__ = ["l_diversity"]


def l_diversity(X, quasi_ids, sensitive, l, c=1.0):
    """Distinct, entropy and recursive (c, l)-diversity of a release.

    Parameters
    ----------
    X : array-like
        Records; only its length is used, to check against the blocks.
    quasi_ids : 2-D array-like
        Quasi-identifier block, one row per record.
    sensitive : array-like
        Sensitive attribute value per record; compared as labels, so
        strings and numbers both work.
    l : int
        Required diversity level, at least 1.
    c : float
        Constant of recursive (c, l)-diversity, strictly positive.

    Returns
    -------
    result : dict
        Keys: estimate (achieved distinct level), distinct_l,
        entropy_l, min_entropy, c_min, satisfies_distinct,
        satisfies_entropy, satisfies_recursive, n_blocks,
        min_block_size, l, c, n.

    References
    ----------
    Machanavajjhala, Kifer, Gehrke & Venkitasubramaniam (2007), ACM TKDD
    1(1):3, doi:10.1145/1217299.1217302, Definitions 4.1 and 4.2.
    """
    n = len(core.vec(X))
    if n == 0:
        raise ValueError("empty input: X has no records")
    rows = core.mat(quasi_ids)
    if len(rows) != n:
        raise ValueError("quasi_ids and X must have the same length")
    sv = [str(v) for v in sensitive]
    if len(sv) != n:
        raise ValueError("sensitive and X must have the same length")
    ll = int(l)
    if ll < 1:
        raise ValueError("l must be at least 1")
    if not (c > 0.0):
        raise ValueError("c must be strictly positive")
    blocks = {}
    order = []
    for i in range(n):
        key = "|".join("%.12g" % v for v in rows[i])
        if key not in blocks:
            blocks[key] = []
            order.append(key)
        blocks[key].append(sv[i])
    distinct_l = None
    min_ent = None
    c_min = 0.0
    min_size = None
    for key in order:
        vals = blocks[key]
        cnt = {}
        seen = []
        for s in vals:
            if s not in cnt:
                cnt[s] = 0
                seen.append(s)
            cnt[s] += 1
        m = len(seen)
        tot = float(len(vals))
        ent = 0.0
        for s in seen:
            p = cnt[s] / tot
            ent -= p * math.log(p)
        # descending counts; ties broken by first appearance so both
        # language arms scan the same order
        r = sorted((cnt[s] for s in seen), reverse=True)
        tail = sum(r[ll - 1:]) if ll <= m else 0.0
        need = float("inf") if tail <= 0.0 else r[0] / float(tail)
        if c_min < need:
            c_min = need
        if distinct_l is None or m < distinct_l:
            distinct_l = m
        if min_ent is None or ent < min_ent:
            min_ent = ent
        if min_size is None or len(vals) < min_size:
            min_size = len(vals)
    return RichResult(
        title="l-diversity",
        summary_lines=[("blocks", len(order)), ("distinct l", distinct_l)],
        payload={
            "estimate": float(distinct_l),
            "distinct_l": float(distinct_l),
            "entropy_l": math.exp(min_ent),
            "min_entropy": min_ent,
            "c_min": c_min,
            "satisfies_distinct": 1.0 if distinct_l >= ll else 0.0,
            "satisfies_entropy": 1.0 if min_ent >= math.log(ll) - 1e-12 else 0.0,
            "satisfies_recursive": 1.0 if (ll == 1 or c_min < c) else 0.0,
            "n_blocks": float(len(order)),
            "min_block_size": float(min_size),
            "l": float(ll),
            "c": float(c),
            "n": n,
            "method": "l-diversity baseline",
        },
    )


def cheatsheet():
    return "dpld: l-diversity baseline"


# compact alias per ledger/NAMING.md
ldiversity = l_diversity
