"""Chinese remainder theorem solver (Sun Tzu, 4th c.; Stein 2009, Thm 2.2.2)."""

from ._richresult import RichResult

__all__ = ["crtT", "chinese_remainder"]


def _egcd(a, b):
    # extended Euclid: returns (g, c, d) with c*a + d*b = g
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def crtT(residues, moduli):
    """
    Solve a system of simultaneous congruences x = a_i (mod m_i).

    The Chinese Remainder Theorem (Stein 2009, Thm. 2.2.2): for
    pairwise coprime moduli a solution exists and is unique modulo
    the product of the moduli.  Solved by folding pairs with Stein's
    Algorithm 2.2.3: with c m + d n = 1 from the extended Euclidean
    algorithm, x = a + (b - a) c m satisfies x = a (mod m) and
    x = b (mod n).  Exact integer arithmetic throughout.  The
    4th-century problem of Sun Tzu (Stein, Question 2.2.1: remainder
    2 by 3, 3 by 5, 2 by 7) is the canonical test case.

    Sources
    -------
    Stein, W. (2009). *Elementary Number Theory: Primes,
    Congruences, and Secrets*, Springer, Sec. 2.2, Question 2.2.1,
    Theorem 2.2.2, Algorithm 2.2.3 (local copy
    fetched-wave3/stein-2009-elementary-number-theory.pdf).
    Sun Tzu (Sunzi Suanjing, 4th c.), as quoted by Stein.
    Knuth, D. E. (1997). *The Art of Computer Programming*, Vol. 2,
    Sec. 4.3.2 (CRT in computer arithmetic, as cited by the stub).

    Parameters
    ----------
    residues : sequence of int
        Values a_i.
    moduli : sequence of int
        Pairwise coprime moduli m_i (each >= 2).

    Returns
    -------
    RichResult
        Keys: estimate (the least non-negative solution), modulus
        (product of moduli), residues, moduli.
    """
    a = [int(v) for v in residues]
    m = [int(v) for v in moduli]
    if len(a) != len(m) or not a:
        raise ValueError("residues and moduli must be non-empty and paired")
    if any(v < 2 for v in m):
        raise ValueError("moduli must be >= 2")
    x = a[0] % m[0]
    mod = m[0]
    for ai, mi in zip(a[1:], m[1:]):
        g, c, _ = _egcd(mod, mi)
        if g != 1:
            raise ValueError("moduli must be pairwise coprime "
                             "(gcd(%d, %d) = %d)" % (mod, mi, g))
        # Algorithm 2.2.3: x_new = x + (b - x) c mod, with c mod + d mi = 1
        x = (x + (ai - x) * c * mod) % (mod * mi)
        mod *= mi
    return RichResult(payload={
        "estimate": x,
        "modulus": mod,
        "residues": a,
        "moduli": m,
        "method": "Chinese remainder theorem (Stein Alg. 2.2.3)",
    })


# long descriptive alias (stub-era name)
chinese_remainder = crtT


def cheatsheet():
    return "crtT: fold pairs via x + (b-x)*c*m with cm+dn=1 (ext. Euclid)"
