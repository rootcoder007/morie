"""Link prediction scores: common neighbours, Adamic-Adar, resource allocation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["linkpr", "link_prediction"]

_METHODS = ("cn", "aa", "ra")


def linkpr(G, u, v, method="all"):
    """
    Neighbourhood link-prediction scores for a node pair (u, v).

    With Gamma(x) the neighbour set of x (nonzero entries of row x,
    self excluded):

        common neighbours    CN(u, v) = |Gamma(u) & Gamma(v)|
        Adamic-Adar          AA(u, v) = sum_{z in Gamma(u) & Gamma(v)} 1 / log |Gamma(z)|
        resource allocation  RA(u, v) = sum_{z in Gamma(u) & Gamma(v)} 1 / |Gamma(z)|

    The graph is treated as unweighted and undirected: any nonzero
    A_ij is an edge.

    Sources
    -------
    Liben-Nowell, D. & Kleinberg, J. (2007). The link-prediction problem
    for social networks. *JASIST*, 58(7), 1019-1031, Sec. 2 ("Methods
    for link prediction": common neighbors score |Gamma(x) & Gamma(y)|;
    Adamic/Adar score sum 1/log|Gamma(z)|)
    (fetched-wave3/libennowell-kleinberg-2007-link-prediction.pdf).
    Zhou, T., Lu, L. & Zhang, Y.-C. (2009). Predicting missing links via
    local information. *European Physical Journal B*, 71, 623-630
    (resource-allocation index, eq. 2: sum over common neighbours of
    1/k_z) (fetched-wave3/zhou-2009-resource-allocation-link-prediction.pdf).

    Parameters
    ----------
    G : array-like, (n, n)
        Adjacency matrix.
    u, v : int
        Node indices (0-based).
    method : str
        One of "cn", "aa", "ra", or "all".

    Returns
    -------
    RichResult
        Keys: cn, aa, ra (requested subset), common_neighbours (indices),
        estimate (the requested score; the CN score when method="all").
    """
    A = np.atleast_2d(np.asarray(G, dtype=float))
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("G must be square")
    u = int(u)
    v = int(v)
    if not (0 <= u < n and 0 <= v < n):
        raise ValueError("u, v must be valid node indices")
    method = str(method).lower()
    if method not in _METHODS + ("all",):
        raise ValueError("method must be one of cn, aa, ra, all")
    nbr = [set(j for j in range(n) if j != i and A[i, j] != 0.0) for i in range(n)]
    common = sorted(nbr[u] & nbr[v])
    deg = [len(nbr[i]) for i in range(n)]
    cn = float(len(common))
    aa = 0.0
    ra = 0.0
    for z in common:
        if deg[z] > 1:
            aa += 1.0 / np.log(float(deg[z]))
        # degree-1 common neighbour is impossible (z touches both u and v)
        ra += 1.0 / float(deg[z])
    scores = {"cn": cn, "aa": float(aa), "ra": float(ra)}
    payload = {"common_neighbours": common, "u": u, "v": v, "n": int(n),
               "method": "Liben-Nowell-Kleinberg CN/AA + Zhou RA link prediction"}
    if method == "all":
        payload.update(scores)
        payload["estimate"] = cn
    else:
        payload[method] = scores[method]
        payload["estimate"] = scores[method]
    return RichResult(payload=payload)


# long descriptive alias (stub-era name)
link_prediction = linkpr


def cheatsheet():
    return "linkpr: CN / Adamic-Adar / resource-allocation link prediction scores"

# public names resolved by fn/_lazy_map.json
linkprediction = linkpr
