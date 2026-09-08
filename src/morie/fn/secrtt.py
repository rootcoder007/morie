# morie.fn -- function file (rootcoder007/morie)
r"""Envelope encryption: a data key per row, a key-encrypting key
above it.

Encrypting a table under one key is simple and has two failure modes
that cannot both be fixed at that level. **Rotation** means
re-encrypting every row, so it is deferred, so it does not happen.
And a **single compromise** exposes everything ever encrypted.

**The envelope splits the problem.** Each record gets its own **data
encryption key** (DEK); the DEK is wrapped under a long-lived
**key-encrypting key** (KEK) that stays in an HSM or KMS and never
appears in the application's memory. The wrapped DEK is stored beside
the ciphertext, so it costs one column, not one lookup.

**Rotation then becomes cheap, and it is worth being exact about what
rotates.** Rotating the KEK re-wraps the DEKs -- one small operation
per record, touching no ciphertext at all. Rotating a DEK does require
re-encrypting that record, so ``rotate_kek`` and ``rotate_dek`` are
separate calls with separate costs, and ``rotation_cost`` compares
them against re-encrypting under a single key.

**What the envelope does and does not contain.** A stolen wrapped DEK
is useless without the KEK, so a database dump alone yields nothing --
this is the property the design exists for. But an attacker who can
*call* the KEK (application-level compromise, not database theft) can
unwrap on demand. The envelope bounds **offline** compromise, not
online abuse, and the honest defence for the latter is the audit trail
of unwrap calls, which is why ``unwrap_dek`` records one.

**Crypto-shredding** follows from the structure: destroy the KEK and
every DEK under it is unrecoverable, which deletes the data without
touching the rows. ``crypto_shred`` reports how many records that
covers, since "we deleted the key" is only a deletion claim if you
know its scope.

References
----------
National Institute of Standards and Technology (2020)
*Recommendation for Key Management: Part 1 - General*, NIST Special
Publication 800-57 Part 1 Revision 5,
doi:10.6028/NIST.SP.800-57pt1r5. The key hierarchy of key-encrypting
keys wrapping data keys, cryptoperiods and the reasons for limiting
them (bounding the amount of information protected under one key and
the exposure from a single compromise), key rotation, and key
destruction as a control.

Housley, R. (2009) "Cryptographic Message Syntax (CMS)", RFC 5652,
doi:10.17487/RFC5652. The enveloped-data structure: content encrypted
under a content-encryption key, with that key wrapped per recipient
and carried alongside the content.

Nir, Y. & Langley, A. (2018) "ChaCha20 and Poly1305 for IETF
Protocols", RFC 8439, doi:10.17487/RFC8439. The AEAD used for both
the wrap and the record; implemented in :mod:`secaead`.

Krawczyk, H. & Eronen, P. (2010) "HKDF", RFC 5869,
doi:10.17487/RFC5869. Per-record key derivation; implemented in
:mod:`seckdf`.
"""

from . import _sha2 as h
from . import secaead as ae
from . import seckdf as kdf
from ._richresult import RichResult

__all__ = ["generate_dek", "wrap_dek", "unwrap_dek", "seal_record",
           "open_record", "rotate_kek", "rotate_dek",
           "rotation_cost", "crypto_shred"]

_EPS = 1e-12


def generate_dek(master_seed, record_id, salt=None):
    r"""A distinct data key per record, derived not stored.

    Deriving from a seed and the record id means the key table is the
    seed plus an identifier, not one row per record.
    """
    r = kdf.hkdf(master_seed, salt, b"dek:" + h._as_bytes(record_id),
                 32)
    return {"dek": r["okm"], "dek_hex": r["okm_hex"],
            "record_id": record_id,
            "note": "per-record, so one leaked DEK exposes one record"}


def wrap_dek(dek, kek, nonce, kek_id="kek-1", aad=b""):
    r"""Encrypt the DEK under the KEK. The result is safe to store."""
    d = h._as_bytes(dek)
    if len(d) != 32:
        raise ValueError("secrtt: a DEK must be 32 bytes, got %d"
                         % len(d))
    bound = h._as_bytes(aad) + h._as_bytes(kek_id)
    r = ae.aead_encrypt(kek, nonce, d, bound)
    return {"wrapped": r["ciphertext"], "tag": r["tag"],
            "nonce": h._as_bytes(nonce), "kek_id": kek_id,
            "wrapped_hex": r["ciphertext_hex"],
            "note": "the KEK id is authenticated, so a wrapped DEK "
                    "cannot be replayed under a different KEK"}


def unwrap_dek(wrapped, kek, audit_log=None):
    r"""Unwrap, and RECORD the call.

    The envelope bounds offline compromise; an attacker who can call
    the KEK is bounded only by what the audit trail catches, so the
    call is logged rather than left silent.
    """
    r = ae.aead_decrypt(kek, wrapped["nonce"], wrapped["wrapped"],
                        wrapped["tag"],
                        h._as_bytes(wrapped.get("aad", b""))
                        + h._as_bytes(wrapped["kek_id"]))
    if audit_log is not None:
        audit_log.append({"event": "unwrap",
                          "kek_id": wrapped["kek_id"],
                          "ok": r["valid"]})
    if not r["valid"]:
        raise ValueError("secrtt: the wrapped DEK failed "
                         "authentication -- wrong KEK, or it was "
                         "tampered with")
    return {"dek": r["plaintext"], "kek_id": wrapped["kek_id"],
            "audited": audit_log is not None}


def seal_record(plaintext, dek, nonce, aad=b""):
    r"""Encrypt one record under its own DEK."""
    r = ae.aead_encrypt(dek, nonce, plaintext, aad)
    return {"ciphertext": r["ciphertext"], "tag": r["tag"],
            "nonce": h._as_bytes(nonce), "aad": h._as_bytes(aad)}


def open_record(sealed, dek):
    r"""Decrypt one record; fails closed."""
    r = ae.aead_decrypt(dek, sealed["nonce"], sealed["ciphertext"],
                        sealed["tag"], sealed.get("aad", b""))
    if not r["valid"]:
        raise ValueError("secrtt: the record failed authentication")
    return r["plaintext"]


def rotate_kek(wrapped_deks, old_kek, new_kek, new_nonces,
               new_kek_id="kek-2", audit_log=None):
    r"""Re-wrap every DEK. NO record ciphertext is touched.

    The cheap rotation, and the reason the envelope exists.
    """
    if len(new_nonces) != len(wrapped_deks):
        raise ValueError("secrtt: %d wrapped DEKs but %d nonces -- a "
                         "nonce must never be reused under a new KEK"
                         % (len(wrapped_deks), len(new_nonces)))
    out = []
    for i, w in enumerate(wrapped_deks):
        dek = unwrap_dek(w, old_kek, audit_log)["dek"]
        out.append(wrap_dek(dek, new_kek, new_nonces[i],
                            new_kek_id))
    return RichResult(payload={
        "estimate": out, "wrapped": out, "n": len(out),
        "records_reencrypted": 0, "kek_id": new_kek_id,
        "method": "envelope KEK rotation; NIST SP 800-57 Part 1 "
                  "Rev. 5",
        "note": "one small re-wrap per record and zero record "
                "ciphertext rewritten",
    })


def rotate_dek(sealed, old_dek, new_dek, new_nonce):
    r"""The expensive rotation: this one DOES re-encrypt the record."""
    pt = open_record(sealed, old_dek)
    return {"sealed": seal_record(pt, new_dek, new_nonce,
                                  sealed.get("aad", b"")),
            "records_reencrypted": 1,
            "note": "rotating a DEK rewrites its record; rotating the "
                    "KEK does not"}


def rotation_cost(n_records, mean_record_bytes, dek_bytes=32):
    r"""Bytes re-encrypted under each strategy."""
    n = int(n_records)
    b = float(mean_record_bytes)
    if n < 1 or b <= 0.0:
        raise ValueError("secrtt: the record count and size must be "
                         "positive")
    single = n * b
    kek = n * float(dek_bytes)
    return {"single_key_bytes": single, "envelope_kek_bytes": kek,
            "ratio": single / kek if kek > 0 else float("inf"),
            "records_touched_single": n,
            "records_touched_envelope": 0,
            "note": "rotation that is expensive is rotation that is "
                    "deferred"}


def crypto_shred(kek_id, wrapped_deks):
    r"""Destroy the KEK; every DEK under it is unrecoverable.

    Deletion without touching the rows -- but the SCOPE has to be
    stated, or 'we destroyed the key' is not a deletion claim.
    """
    covered = [i for i, w in enumerate(wrapped_deks)
               if w["kek_id"] == kek_id]
    orphaned = [i for i, w in enumerate(wrapped_deks)
                if w["kek_id"] != kek_id]
    return {"kek_id": kek_id, "records_shredded": len(covered),
            "indices": covered,
            "still_recoverable": orphaned,
            "complete": not orphaned,
            "note": "any DEK wrapped under a DIFFERENT KEK survives, "
                    "so a partial rotation leaves data readable"}


def cheatsheet():
    return ("secrtt: one key for a whole table means rotation "
            "re-encrypts everything (so it is deferred, so it never "
            "happens) and one compromise exposes everything. ENVELOPE: "
            "a DEK per record, wrapped under a long-lived KEK held in "
            "an HSM, stored beside the ciphertext. Rotating the KEK "
            "re-wraps DEKs and touches NO record ciphertext; rotating "
            "a DEK does re-encrypt its record -- different costs, "
            "different calls. The envelope bounds OFFLINE compromise: "
            "a stolen database is useless, but an attacker who can "
            "CALL the KEK unwraps on demand, so log every unwrap. "
            "CRYPTO-SHREDDING deletes by destroying the KEK -- state "
            "the scope, since DEKs under another KEK survive.")


# compact alias per ledger/NAMING.md
rotating_token_envelope = rotate_kek
