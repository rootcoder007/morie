# morie.fn -- function file (rootcoder007/morie)
"""Rice index of party cohesion on roll calls."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rice_index"]


def rice_index(votes, party_id):
    r"""Rice cohesion per party and roll call.

    .. math:: \mathrm{Rice}_{pj} = \big| \%\text{yea}_{pj}
              - \%\text{nay}_{pj} \big| \in [0, 1],

    computed over the party's non-missing votes on roll call j; 1 is a
    unanimous party, 0 an evenly split one. Party means are averaged
    over roll calls where the party cast at least one vote.

    Parameters
    ----------
    votes : array-like, shape (n, q)
        Binary roll-call matrix (1 = yea, 0 = nay, NaN = missing).
    party_id : array-like, shape (n,)
        Party label per legislator.

    Returns
    -------
    RichResult
        keys: ``by_party`` (dict party -> mean Rice), ``matrix``
        (dict party -> per-roll-call array), ``parties``, ``n``,
        ``method``.

    References
    ----------
    Rice, S. A. (1925). The behavior of legislative groups: a method
    of measurement. *Political Science Quarterly*, 40(1), 60-72.
    """
    V = np.asarray(votes, dtype=float)
    if V.ndim != 2:
        raise ValueError("votes must be 2-D.")
    pid = np.asarray(party_id).ravel()
    if pid.size != V.shape[0]:
        raise ValueError("party_id must have one entry per legislator.")

    parties = [p for p in dict.fromkeys(pid.tolist())]
    per = {}
    means = {}
    for p in parties:
        rows = V[pid == p]
        vals = []
        for j in range(V.shape[1]):
            col = rows[:, j]
            col = col[~np.isnan(col)]
            if col.size == 0:
                vals.append(np.nan)
                continue
            yea = col.mean()
            vals.append(abs(yea - (1 - yea)))
        arr = np.array(vals)
        per[p] = arr
        means[p] = float(np.nanmean(arr)) if np.any(~np.isnan(arr)) else float("nan")

    return RichResult(
        payload={
            "by_party": means,
            "matrix": per,
            "parties": parties,
            "n": int(V.shape[0]),
            "method": "Rice index |%yea - %nay| per party and roll call",
        }
    )


def cheatsheet():
    return "ricei: Rice_pj = |%yea - %nay| within party p on roll call j (Rice 1925)"
