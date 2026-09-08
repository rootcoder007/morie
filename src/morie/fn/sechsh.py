# morie.fn -- function file (rootcoder007/morie)
r"""A hash-chained audit log: tamper-evident, not tamper-proof.

An audit log that an administrator can edit is not evidence. Chaining
each entry to its predecessor,

.. math:: h_i = H(h_{i-1} \,\|\, \mathrm{entry}_i),

makes any edit to entry :math:`j` change :math:`h_j` and therefore
every hash after it, so **verification localises the damage**: the
chain is intact up to the first mismatch and unverifiable beyond it.
``verify_chain`` returns that index rather than a bare boolean,
because "the log is broken" and "the log is broken from entry 412"
are different operational facts.

**Be precise about what this does and does not buy.** A plain hash
chain is tamper-**evident**: it detects edits and reordering. It does
not stop an attacker who can rewrite the *whole* chain from the point
of compromise onward -- they simply recompute every later hash. Two
standard defences, both here:

* **Keyed chaining** (:math:`\mathrm{HMAC}(k, h_{i-1}\|e_i)`) with a
  key the log writer does not hold, so forward rewriting needs the
  key as well as write access.
* **Anchoring** the head hash somewhere outside the system's control,
  which bounds any rewrite to entries after the last anchor.

**Merkle inclusion proofs** answer a different question: given a
trusted head, prove that one entry is in the log without sending the
whole log. The proof is :math:`\log_2 n` hashes, and the audit path
construction here follows RFC 6962's, including its leaf/interior
domain separation -- prefixing 0x00 to leaves and 0x01 to internal
nodes, without which an interior node could be presented as a leaf.

References
----------
Laurie, B., Langley, A. & Kasper, E. (2013) "Certificate
Transparency", RFC 6962, doi:10.17487/RFC6962. Sec. 2: the Merkle
Tree Hash with MTH({}) = HASH() of the empty string, leaf hashing as
HASH(0x00 || d) and interior nodes as HASH(0x01 || left || right) --
the domain separation that prevents a node being reinterpreted --
the split of a list at the largest power of two strictly less than
its length, and the audit path proving inclusion of a leaf against a
known tree head.

Schneier, B. & Kelsey, J. (1999) "Secure Audit Logs to Support
Computer Forensics", *ACM Transactions on Information and System
Security* 2(2), 159-176, doi:10.1145/317087.317089. The
forward-integrity argument: hash chaining makes a log tamper-evident,
and a key the logging machine cannot recover is what stops an
attacker who compromises it from rewriting earlier entries
undetectably.

National Institute of Standards and Technology (2015) *Secure Hash
Standard (SHS)*, FIPS PUB 180-4, doi:10.6028/NIST.FIPS.180-4. The
hash; implemented in :mod:`_sha2`.
"""

from . import _sha2 as h
from ._richresult import RichResult

__all__ = ["chain_entry", "build_chain", "verify_chain",
           "merkle_root", "inclusion_proof", "verify_inclusion"]

_LEAF = b"\x00"
_NODE = b"\x01"
GENESIS = b"\x00" * 32


def chain_entry(previous_hash, entry, key=None):
    r""":math:`h_i = H(h_{i-1}\|e_i)`, or HMAC if a key is given.

    With a key the log writer does not hold, an attacker with write
    access still cannot recompute the chain forward.
    """
    p = h._as_bytes(previous_hash)
    e = h._as_bytes(entry)
    if key is None:
        return {"hash": h.sha256(p + e), "keyed": False}
    return {"hash": h.hmac_sha256(key, p + e), "keyed": True,
            "note": "forward rewriting now needs the KEY as well as "
                    "write access"}


def build_chain(entries, key=None, genesis=GENESIS):
    r"""Chain a list of entries, returning every intermediate hash."""
    prev = h._as_bytes(genesis)
    hashes = []
    for e in entries:
        prev = chain_entry(prev, e, key)["hash"]
        hashes.append(prev)
    return {"hashes": hashes, "head": prev if hashes else
            h._as_bytes(genesis), "n": len(hashes),
            "head_hex": h.hexlify(prev if hashes else genesis),
            "keyed": key is not None}


def verify_chain(entries, hashes, key=None, genesis=GENESIS):
    r"""Where does the chain first fail?

    Returning the index matters: everything before it is still
    evidence, and everything after it is not.
    """
    if len(entries) != len(hashes):
        raise ValueError("sechsh: %d entries but %d hashes -- an "
                         "entry or a hash has been dropped"
                         % (len(entries), len(hashes)))
    prev = h._as_bytes(genesis)
    first_bad = None
    for i in range(len(entries)):
        want = chain_entry(prev, entries[i], key)["hash"]
        if not h.constant_time_equal(want, hashes[i]):
            if first_bad is None:
                first_bad = i
        prev = h._as_bytes(hashes[i])
    return RichResult(payload={
        "estimate": first_bad is None, "intact": first_bad is None,
        "first_bad": first_bad,
        "verified_through": len(entries) if first_bad is None
        else first_bad,
        "n": len(entries),
        "method": "hash-chained audit log; Schneier & Kelsey (1999)",
        "note": "tamper-EVIDENT, not tamper-proof: an attacker who "
                "can rewrite the whole tail recomputes every later "
                "hash, which is what keying and external anchoring "
                "are for",
    })


def merkle_root(leaves):
    r"""RFC 6962 Merkle Tree Hash.

    Leaves are prefixed 0x00 and interior nodes 0x01, so a node can
    never be presented as a leaf; the split is at the largest power
    of two strictly less than the length.
    """
    L = [h._as_bytes(v) for v in leaves]
    if not L:
        return h.sha256(b"")
    if len(L) == 1:
        return h.sha256(_LEAF + L[0])
    k = 1
    while k * 2 < len(L):
        k *= 2
    return h.sha256(_NODE + merkle_root(L[:k]) + merkle_root(L[k:]))


def inclusion_proof(leaves, index):
    r"""The audit path: :math:`\log_2 n` hashes, not the whole log."""
    L = [h._as_bytes(v) for v in leaves]
    m = int(index)
    if m < 0 or m >= len(L):
        raise ValueError("sechsh: index %d is outside a log of %d"
                         % (m, len(L)))
    path = []
    lo, hi = 0, len(L)
    while hi - lo > 1:
        k = 1
        while k * 2 < hi - lo:
            k *= 2
        if m - lo < k:
            path.append(merkle_root(L[lo + k:hi]))
            hi = lo + k
        else:
            path.append(merkle_root(L[lo:lo + k]))
            lo = lo + k
    return {"path": path, "path_hex": [h.hexlify(v) for v in path],
            "length": len(path), "index": m, "size": len(L),
            "note": "log2(n) hashes prove membership against a "
                    "trusted head"}


def verify_inclusion(leaf, index, size, path, root):
    r"""Recompute the head from the leaf and the path alone."""
    m, n = int(index), int(size)
    if m < 0 or m >= n:
        raise ValueError("sechsh: index %d is outside a log of %d"
                         % (m, n))
    node = h.sha256(_LEAF + h._as_bytes(leaf))
    # The audit path is recorded top-down by inclusion_proof, but the
    # hashing has to run bottom-up from the leaf: collect the descent
    # first, then fold it in reverse. Consuming the path in the order
    # it was written combines the wrong sibling at the wrong level,
    # and on a tree whose leaf count is not a power of two -- where
    # the depths differ per leaf -- that shows up immediately.
    lo, hi = 0, n
    steps, used = [], 0
    p = list(path)
    while hi - lo > 1:
        if used >= len(p):
            raise ValueError("sechsh: the audit path is too short "
                             "for a log of %d" % n)
        k = 1
        while k * 2 < hi - lo:
            k *= 2
        sib = h._as_bytes(p[used])
        used += 1
        if m - lo < k:
            steps.append((sib, True))
            hi = lo + k
        else:
            steps.append((sib, False))
            lo = lo + k
    for sib, on_right in reversed(steps):
        node = (h.sha256(_NODE + node + sib) if on_right
                else h.sha256(_NODE + sib + node))
    return {"root": node, "root_hex": h.hexlify(node),
            "valid": h.constant_time_equal(node, root),
            "path_used": used}


def cheatsheet():
    return ("sechsh: h_i = H(h_{i-1} || entry_i) makes a log "
            "TAMPER-EVIDENT -- any edit changes every later hash, so "
            "verification localises the damage and should return the "
            "FIRST BAD INDEX, not a boolean. It is NOT tamper-proof: "
            "an attacker who rewrites the whole tail just recomputes "
            "the hashes. Two defences: KEYED chaining with a key the "
            "writer does not hold, and ANCHORING the head outside the "
            "system, which bounds any rewrite to entries after the "
            "last anchor. Merkle inclusion proves membership in "
            "log2(n) hashes; prefix 0x00 to leaves and 0x01 to "
            "interior nodes or a node can be passed off as a leaf.")


# compact alias per ledger/NAMING.md
hash_chained_log = verify_chain

# public names resolved by fn/_lazy_map.json
hash_chain_audit = verify_chain
hashchainaudit = verify_chain
