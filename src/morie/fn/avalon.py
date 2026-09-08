"""The Avalon fingerprint: a molecule reduced to a bit vector by
enumerating structural features and hashing each one into a bit.

A fingerprint is a lossy summary with one job -- two molecules that
share substructure must share bits, so that a similarity computed on
the bits stands in for a similarity computed on the graphs. Avalon's
approach, as described by Gedeck, Rohde and Bartels, is to enumerate a
fixed catalogue of query features rather than to grow environments
around every atom the way a Morgan fingerprint does, and to fold the
catalogue into a bit vector of the caller's chosen width.

WHAT IS FAITHFUL AND WHAT IS THIS MODULE'S OWN. The paper describes the
fingerprint's design and evaluates it; the exact feature enumeration
and the hash seeds that assign a feature to a bit live in the Avalon
toolkit's C source, not in the paper. So the bit POSITIONS here will
not agree with the Avalon toolkit's, and this module does not claim
they do. What is implemented is the fingerprint's structure -- five
named feature classes, each hashed with its own seed -- and its
contract, which is what a fingerprint is actually used for and what the
anchors below test:

  ``atom``  element, aromaticity, formal charge, degree and ring
            membership. The smallest feature there is.
  ``bond``  the two atom types and the bond order, written with the two
            ends in a fixed order so a bond and its reverse are one
            feature and not two.
  ``path``  every simple path up to a length limit, as an alternating
            sequence of atom types and bond orders, again canonicalised
            against its own reverse.
  ``ring``  ring size, whether the ring is aromatic, and the sorted
            elements around it.
  ``pair``  Carhart's atom pairs: two atom types and the number of
            bonds between them.

The classes are a parameter, and they do not all behave the same way.
The PATH and BOND classes are SUBGRAPH MONOTONE -- every path or bond
of a fragment is a path or bond of anything containing it -- which is
what lets a fingerprint be used as a substructure screen. The ATOM
class is not, because an atom feature carries the atom's degree and
hydrogen count, and those are properties of the whole molecule: propane
has a carbon of degree two, and isobutane, which contains propane, has
none. Neither are the RING and PAIR classes, since a ring or a distance
can appear only once the rest of the molecule is there. That
distinction is real and it is anchored in both directions rather than
glossed.

THE PARSER. SMILES is parsed here rather than assumed: the organic
subset, bracket atoms with isotope, charge and explicit hydrogen count,
bond symbols, branches, ring-closure digits and percent labels, and the
dot disconnection. Anything outside that is REFUSED with a message
naming what was found, because a parser that silently drops what it
does not understand produces a fingerprint of a different molecule.

Implicit hydrogens follow the standard valences of the organic subset,
with an aromatic atom counted as carrying one extra bond -- which is
what makes each carbon of benzene come out with exactly one hydrogen.

Rings are found from the ring-closure bonds: the smallest ring through
a closure bond is that bond plus the shortest path between its ends
that avoids it. For a fused system this is a valid ring set but not
necessarily the canonical smallest set of smallest rings, and it is
described as what it is.

References
  Gedeck, P., Rohde, B. and Bartels, C. (2006) "QSAR -- how good is it
    in practice? Comparison of descriptor sets on an unbiased cross
    section of database candidates." Journal of Chemical Information
    and Modeling 46(5), 1924-1936. doi:10.1021/ci050413p. Where the
    Avalon fingerprint is introduced and benchmarked.
  Carhart, R.E., Smith, D.H. and Venkataraghavan, R. (1985) "Atom pairs
    as molecular features in structure-activity studies: definition and
    applications." Journal of Chemical Information and Computer
    Sciences 25(2), 64-73. The pair feature class.
  Weininger, D. (1988) "SMILES, a chemical language and information
    system. 1. Introduction to methodology and encoding rules."
    Journal of Chemical Information and Computer Sciences 28(1), 31-36.
  Fowler, G., Noll, L.C. and Vo, K.-P. FNV-1a, the 32-bit
    multiply-and-xor hash used to fold a feature into a bit. Chosen
    because it is fully specified in integer arithmetic, so both arms
    of this package compute the same bit rather than the same
    approximate bit.
"""

from ._richresult import RichResult

__all__ = ["avalon_fingerprint", "parse_smiles", "ring_bonds",
           "tanimoto", "features_of", "cheatsheet"]

# The organic subset: elements that may be written without brackets,
# and the valence each is filled up to when counting implicit hydrogens.
_ORGANIC = {"B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2,
            "F": 1, "Cl": 1, "Br": 1, "I": 1}
_AROMATIC = {"b": "B", "c": "C", "n": "N", "o": "O", "p": "P", "s": "S"}
_BONDS = {"-": 1, "=": 2, "#": 3, ":": 4, "/": 1, "\\": 1}
_CLASSES = ("atom", "bond", "path", "ring", "pair")


def _fnv(s, seed=2166136261):
    """FNV-1a over the bytes of a feature key.

    Written as explicit modular arithmetic rather than with a language's
    integer type, so the R arm computes the same number instead of a
    number that rounds the same way.
    """
    h = seed
    for ch in s:
        b = ord(ch)
        if b > 255:
            raise ValueError("a feature key must be plain ASCII")
        lo = h % 256
        h = h - lo + (int(lo) ^ b)
        hi = h // 65536
        low = h % 65536
        h = ((hi * 16777619) % 65536) * 65536 + low * 16777619
        h = h % 4294967296
    return h


def parse_smiles(smiles):
    """A SMILES string as an atom-bond graph.

    Returns the element of each atom, whether it was written aromatic,
    its formal charge, its explicit hydrogen count if one was given, and
    the bonds as index triples with the order. Ring-closure bonds are
    reported separately because the ring perception below starts from
    them.
    """
    s = str(smiles)
    el = []
    arom = []
    chg = []
    hexp = []
    bonds = []
    closures = []
    open_ring = {}
    stack = []
    prev = -1
    order = 0
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "(":
            if prev < 0:
                raise ValueError("a branch cannot open before an atom")
            stack.append(prev)
            i += 1
            continue
        if ch == ")":
            if not stack:
                raise ValueError("a branch closed that was never opened")
            prev = stack.pop()
            i += 1
            continue
        if ch in _BONDS:
            order = _BONDS[ch]
            i += 1
            continue
        if ch == ".":
            prev = -1
            order = 0
            i += 1
            continue
        if ch == "%" or ch.isdigit():
            if ch == "%":
                if i + 2 >= n or not s[i + 1:i + 3].isdigit():
                    raise ValueError("a percent ring label needs two "
                                     "digits")
                lab = s[i + 1:i + 3]
                i += 3
            else:
                lab = ch
                i += 1
            if prev < 0:
                raise ValueError("a ring closure cannot precede an atom")
            if lab in open_ring:
                a, o = open_ring.pop(lab)
                oo = order if order else (o if o else 1)
                bonds.append((a, prev, oo))
                closures.append(len(bonds) - 1)
            else:
                open_ring[lab] = (prev, order)
            order = 0
            continue
        # An atom, bracketed or not.
        if ch == "[":
            j = s.find("]", i)
            if j < 0:
                raise ValueError("a bracket atom was never closed")
            body = s[i + 1:j]
            i = j + 1
            k = 0
            # A leading isotope number is read and discarded: mass is not
            # one of the feature classes, so keeping it would put two
            # bits where the fingerprint defines one.
            while k < len(body) and body[k].isdigit():
                k += 1
            sym = ""
            if k < len(body) and body[k].isalpha():
                sym = body[k]
                k += 1
                if k < len(body) and body[k].islower() \
                        and (sym + body[k]) in _ORGANIC:
                    sym = sym + body[k]
                    k += 1
            if not sym:
                raise ValueError("a bracket atom must name an element")
            ar = sym in _AROMATIC
            e = _AROMATIC[sym] if ar else sym
            hh = -1
            cg = 0
            while k < len(body):
                c = body[k]
                if c == "H":
                    k += 1
                    d = ""
                    while k < len(body) and body[k].isdigit():
                        d += body[k]
                        k += 1
                    hh = int(d) if d else 1
                elif c in "+-":
                    sgn = 1 if c == "+" else -1
                    k += 1
                    d = ""
                    while k < len(body) and body[k].isdigit():
                        d += body[k]
                        k += 1
                    if d:
                        cg = sgn * int(d)
                    else:
                        cg = sgn
                        while k < len(body) and body[k] == c:
                            cg += sgn
                            k += 1
                elif c == "@":
                    # Stereochemistry is not a feature of this
                    # fingerprint, and dropping it is a decision, not an
                    # oversight: two enantiomers get the same bits.
                    k += 1
                    while k < len(body) and body[k] == "@":
                        k += 1
                else:
                    raise ValueError("unsupported bracket atom field: "
                                     + c)
        else:
            if i + 1 < n and s[i:i + 2] in _ORGANIC:
                sym = s[i:i + 2]
                i += 2
            elif ch in _ORGANIC or ch in _AROMATIC:
                sym = ch
                i += 1
            else:
                raise ValueError("unsupported SMILES character: " + ch)
            ar = sym in _AROMATIC
            e = _AROMATIC[sym] if ar else sym
            hh = -1
            cg = 0
        el.append(e)
        arom.append(1 if ar else 0)
        chg.append(cg)
        hexp.append(hh)
        cur = len(el) - 1
        if prev >= 0:
            oo = order if order else (4 if (arom[prev] and ar) else 1)
            bonds.append((prev, cur, oo))
        prev = cur
        order = 0
    if open_ring:
        raise ValueError("a ring closure was opened and never matched")
    if stack:
        raise ValueError("a branch was opened and never closed")
    if not el:
        raise ValueError("an empty SMILES string is not a molecule")
    return el, arom, chg, hexp, bonds, closures


def _adjacency(n, bonds):
    adj = [[] for _ in range(n)]
    for k in range(len(bonds)):
        a, b, o = bonds[k]
        adj[a].append((b, o, k))
        adj[b].append((a, o, k))
    return adj


def implicit_h(el, arom, chg, hexp, bonds):
    """Hydrogens filled in to the standard valence of the organic subset.

    An aromatic atom is counted as carrying one extra bond, which is
    what makes each carbon of benzene come out with exactly one
    hydrogen. An explicit count in brackets is taken as given and
    nothing is filled in.
    """
    n = len(el)
    used = [0] * n
    for a, b, o in bonds:
        w = 1 if o == 4 else o
        used[a] += w
        used[b] += w
    out = []
    for i in range(n):
        if hexp[i] >= 0:
            out.append(hexp[i])
            continue
        v = _ORGANIC.get(el[i])
        if v is None:
            out.append(0)
            continue
        need = v + chg[i] - used[i] - (1 if arom[i] else 0)
        out.append(need if need > 0 else 0)
    return out


def _shortest(adj, src, dst, banned):
    """Fewest bonds from src to dst without using the banned bond."""
    n = len(adj)
    dist = [-1] * n
    prev = [-1] * n
    dist[src] = 0
    q = [src]
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v, o, k in adj[u]:
            if k == banned or dist[v] >= 0:
                continue
            dist[v] = dist[u] + 1
            prev[v] = u
            q.append(v)
    if dist[dst] < 0:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def ring_bonds(n, bonds, closures):
    """One ring per ring-closure bond: that bond plus the shortest path.

    For a single ring this is the ring. For a fused system it is a valid
    ring set of the right size -- the cyclomatic number -- but not
    necessarily the canonical smallest set of smallest rings, and it is
    reported as what it is rather than labelled SSSR.
    """
    adj = _adjacency(n, bonds)
    rings = []
    inring = [0] * n
    for k in closures:
        a, b, o = bonds[k]
        p = _shortest(adj, a, b, k)
        if p is None:
            continue
        rings.append(p)
        for v in p:
            inring[v] = 1
    return rings, inring


def _atype(el, arom, i):
    return (el[i].lower() if arom[i] else el[i])


def _paths(adj, n, maxpath, ty):
    """Every simple path of one to maxpath bonds, each written once.

    A path and its reverse are the same feature, so only the smaller of
    the two spellings is kept -- otherwise a symmetric molecule would
    light twice as many bits as an asymmetric one for no chemical
    reason. The key is built here rather than by the caller so the two
    arms of this package cannot canonicalise it differently.
    """
    out = set()

    def key(seq):
        fwd = []
        rev = []
        for q in range(len(seq)):
            fwd.append(ty[seq[q]] if q % 2 == 0 else str(seq[q]))
        for q in range(len(seq) - 1, -1, -1):
            rev.append(ty[seq[q]] if q % 2 == 0 else str(seq[q]))
        a = "|".join(fwd)
        b = "|".join(rev)
        return "P|" + (a if a <= b else b)

    def walk(seq, used):
        if len(seq) >= 2:
            out.add(key(seq))
        if (len(seq) - 1) // 2 >= maxpath:
            return
        u = seq[-1]
        for v, o, k in adj[u]:
            if v in used:
                continue
            used.add(v)
            seq.append(o)
            seq.append(v)
            walk(seq, used)
            seq.pop()
            seq.pop()
            used.discard(v)

    for s in range(n):
        walk([s], set([s]))
    return out


def _bfs_dist(adj, n):
    D = []
    for s in range(n):
        d = [-1] * n
        d[s] = 0
        q = [s]
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            for v, o, k in adj[u]:
                if d[v] < 0:
                    d[v] = d[u] + 1
                    q.append(v)
        D.append(d)
    return D


def features_of(smiles, maxpath=5, classes=None):
    """The feature keys of a molecule, as sorted strings.

    Exposed because a fingerprint that cannot be asked what it saw is a
    fingerprint that cannot be debugged: the bits are these keys hashed,
    and a surprising bit can always be traced back to a feature here.
    """
    if classes is None:
        classes = _CLASSES
    for c in classes:
        if c not in _CLASSES:
            raise ValueError("unknown feature class: " + str(c))
    el, arom, chg, hexp, bonds, closures = parse_smiles(smiles)
    n = len(el)
    adj = _adjacency(n, bonds)
    rings, inring = ring_bonds(n, bonds, closures)
    nh = implicit_h(el, arom, chg, hexp, bonds)
    ty = [_atype(el, arom, i) for i in range(n)]
    out = set()
    if "atom" in classes:
        for i in range(n):
            out.add("A|%s|%d|%d|%d|%d|%d"
                    % (ty[i], arom[i], chg[i], len(adj[i]), inring[i],
                       nh[i]))
    if "bond" in classes:
        for a, b, o in bonds:
            x, y = ty[a], ty[b]
            if y < x:
                x, y = y, x
            out.add("B|%s|%d|%s" % (x, o, y))
    if "path" in classes:
        for p in _paths(adj, n, maxpath, ty):
            out.add(p)
    if "ring" in classes:
        for r in rings:
            allarom = 1
            for v in r:
                if not arom[v]:
                    allarom = 0
            elems = sorted(el[v] for v in r)
            out.add("R|%d|%d|%s" % (len(r), allarom, ",".join(elems)))
    if "pair" in classes:
        D = _bfs_dist(adj, n)
        for i in range(n):
            for j in range(i + 1, n):
                if D[i][j] < 0:
                    continue
                x, y = ty[i], ty[j]
                if y < x:
                    x, y = y, x
                out.add("D|%s|%s|%d" % (x, y, D[i][j]))
    return sorted(out)


def tanimoto(a, b):
    """The Tanimoto coefficient of two bit vectors of the same width.

    Two empty fingerprints have nothing in common and nothing to
    disagree about; the ratio is zero over zero and is reported as
    zero rather than as a division that happened to not raise.
    """
    if len(a) != len(b):
        raise ValueError("two fingerprints of different widths cannot "
                         "be compared")
    both = 0
    either = 0
    for i in range(len(a)):
        if a[i] and b[i]:
            both += 1
        if a[i] or b[i]:
            either += 1
    return (both / float(either)) if either else 0.0


def avalon_fingerprint(smiles, n_bits=512, maxpath=5, classes=None):
    """Hash a molecule's structural features into a bit vector.

    Parameters
    ----------
    smiles : str
        The molecule, in the SMILES subset the parser accepts.
    n_bits : int
        The width of the vector. Folding is by remainder, so a narrower
        vector is the wider one folded and collisions rise with it --
        which is why the collision count is reported and not hidden.
    maxpath : int
        The longest path feature, in bonds.
    classes : sequence or None
        Which feature classes to use; None is all five. Restricted to
        bond and path the fingerprint is subgraph monotone.

    Returns
    -------
    RichResult
        The bits, the features that set them, and the collision count.

    References
    ----------
    Gedeck et al. (2006) J. Chem. Inf. Model. 46(5), 1924-1936; Carhart
    et al. (1985) J. Chem. Inf. Comput. Sci. 25(2), 64-73.
    """
    n_bits = int(n_bits)
    if n_bits < 1:
        raise ValueError("a fingerprint needs at least one bit")
    feats = features_of(smiles, maxpath, classes)
    bits = [0] * n_bits
    owner = {}
    coll = 0
    for f in feats:
        b = _fnv(f) % n_bits
        if b in owner:
            coll += 1
        else:
            owner[b] = f
        bits[b] = 1
    el, arom, chg, hexp, bonds, closures = parse_smiles(smiles)
    rings, inring = ring_bonds(len(el), bonds, closures)
    return RichResult(payload={
        "bits": bits,
        "on": [i for i in range(n_bits) if bits[i]],
        "features": feats,
        "n_features": len(feats),
        "n_on": sum(bits),
        "n_collisions": coll,
        "density": sum(bits) / float(n_bits),
        "n_atoms": len(el),
        "n_bonds": len(bonds),
        "n_rings": len(rings),
        "n_hydrogens": sum(implicit_h(el, arom, chg, hexp, bonds)),
        "n_bits": n_bits,
        "maxpath": int(maxpath),
        "classes": list(classes) if classes is not None
                   else list(_CLASSES),
        "method": "Avalon-style hashed feature fingerprint",
    })


def cheatsheet():
    return ("avalon: Avalon-style feature fingerprint. Atom, bond, "
            "path, ring and atom-pair features hashed with FNV-1a into "
            "a folded bit vector; SMILES parsed, not assumed")
