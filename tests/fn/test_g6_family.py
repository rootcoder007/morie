"""Tests for secaead (RFC 8439 vectors) and secrtt (envelope)."""
import importlib

import pytest

h = importlib.import_module("morie.fn._sha2")
ae = importlib.import_module("morie.fn.secaead")
rt = importlib.import_module("morie.fn.secrtt")
H, U = h.hexlify, h.unhexlify

KEY32 = U("000102030405060708090a0b0c0d0e0f"
          "101112131415161718191a1b1c1d1e1f")
AEAD_KEY = bytes(bytearray(range(0x80, 0xa0)))
AEAD_NONCE = U("070000004041424344454647")
AEAD_AAD = U("50515253c0c1c2c3c4c5c6c7")
AEAD_PT = (b"Ladies and Gentlemen of the class of '99: If I could "
           b"offer you only one tip for the future, sunscreen would "
           b"be it.")


def test_chacha20_block_rfc8439_2_3_2():
    blk = ae.chacha20_block(KEY32, 1, U("000000090000004a00000000"))
    assert H(blk).startswith("10f1e7e4d13b5915500fdd1fa32071c4")


def test_chacha20_cipher_rfc8439_2_4_2():
    ct = ae.chacha20(KEY32, 1, U("000000000000004a00000000"),
                     AEAD_PT)
    assert H(ct).startswith("6e2e359a2568f98041ba0728dd0d6981")


def test_chacha20_is_its_own_inverse():
    once = ae.chacha20(U("00" * 32), 1, U("00" * 12), b"round trip")
    assert ae.chacha20(U("00" * 32), 1, U("00" * 12),
                       once) == b"round trip"


def test_poly1305_key_gen_rfc8439_2_6_2():
    otk = ae.poly1305_key_gen(AEAD_KEY, U("000000000001020304050607"))
    assert H(otk) == ("8ad5a08b905f81cc815040274ab29471"
                      "a833b637e3fd0da508dbb8e2fdd1a646")


def test_poly1305_mac_rfc8439_2_5_2():
    mac = ae.poly1305_mac(b"Cryptographic Forum Research Group",
                          U("85d6be7857556d337f4452fe42d506a8"
                            "0103808afb0db2fd4abff6af4149f51b"))
    assert H(mac) == "a8061dc1305136c6c22b8baf0c0127a9"


def test_aead_rfc8439_2_8_2_vector():
    r = ae.aead_encrypt(AEAD_KEY, AEAD_NONCE, AEAD_PT, AEAD_AAD)
    assert r["ciphertext_hex"].startswith(
        "d31a8d34648e60db7b86afbc53ef7ec2")
    assert r["tag_hex"] == "1ae10b594f09e26a7e902ecbd0600691"


def test_aead_round_trip_and_tamper_detection():
    r = ae.aead_encrypt(AEAD_KEY, AEAD_NONCE, AEAD_PT, AEAD_AAD)
    good = ae.aead_decrypt(AEAD_KEY, AEAD_NONCE, r["ciphertext"],
                           r["tag"], AEAD_AAD)
    assert good["valid"] and good["plaintext"] == AEAD_PT
    bad = bytearray(r["ciphertext"])
    bad[0] ^= 0x01
    fail = ae.aead_decrypt(AEAD_KEY, AEAD_NONCE, bytes(bad),
                           r["tag"], AEAD_AAD)
    assert not fail["valid"] and fail["plaintext"] is None


def test_aead_authenticates_the_aad():
    r = ae.aead_encrypt(AEAD_KEY, AEAD_NONCE, AEAD_PT, AEAD_AAD)
    out = ae.aead_decrypt(AEAD_KEY, AEAD_NONCE, r["ciphertext"],
                          r["tag"], AEAD_AAD[:-1] + b"\x00")
    assert not out["valid"]


def test_length_block_disambiguates_the_fields():
    otk = ae.poly1305_key_gen(AEAD_KEY, AEAD_NONCE)
    assert ae.poly1305_mac(ae._mac_data(b"ab", b"cd"), otk) != \
        ae.poly1305_mac(ae._mac_data(b"a", b"bcd"), otk)


def test_clamping_clears_the_required_bits():
    r = ae._clamp(int.from_bytes(b"\xff" * 16, "little"))
    for i in (3, 7, 11, 15):
        assert (r >> (8 * i + 4)) & 0xf == 0
    for i in (4, 8, 12):
        assert (r >> (8 * i)) & 0x3 == 0


def test_secaead_rejects_wrong_lengths():
    with pytest.raises(ValueError):
        ae.chacha20_block(b"short", 0, U("00" * 12))
    with pytest.raises(ValueError):
        ae.chacha20_block(U("00" * 32), 0, b"short")
    with pytest.raises(ValueError):
        ae.poly1305_mac(b"m", b"short")


# ----------------------------------------------------------------- secrtt
KEK = b"K" * 32


def test_secrtt_dek_is_per_record_and_reproducible():
    a = rt.generate_dek(b"master seed", b"row-1")
    b = rt.generate_dek(b"master seed", b"row-2")
    assert a["dek"] != b["dek"] and len(a["dek"]) == 32
    assert rt.generate_dek(b"master seed", b"row-1")["dek"] == a["dek"]


def test_secrtt_wrapped_dek_needs_the_kek():
    d = rt.generate_dek(b"seed", b"row-1")["dek"]
    w = rt.wrap_dek(d, KEK, b"nonce-0-0001"[:12])
    log = []
    assert rt.unwrap_dek(w, KEK, log)["dek"] == d
    assert log and log[0]["event"] == "unwrap"
    with pytest.raises(ValueError):
        rt.unwrap_dek(w, b"W" * 32)


def test_secrtt_record_seals_and_fails_closed():
    d1 = rt.generate_dek(b"seed", b"row-1")["dek"]
    d2 = rt.generate_dek(b"seed", b"row-2")["dek"]
    sealed = rt.seal_record(b"salary: 42", d1, b"record-nonce"[:12])
    assert rt.open_record(sealed, d1) == b"salary: 42"
    with pytest.raises(ValueError):
        rt.open_record(sealed, d2)


def test_secrtt_kek_rotation_touches_no_ciphertext():
    d = rt.generate_dek(b"seed", b"row-1")["dek"]
    w = rt.wrap_dek(d, KEK, b"nonce-0-0001"[:12])
    r = rt.rotate_kek([w], KEK, b"L" * 32, [b"new-nonce-01"[:12]],
                      "kek-2")
    assert r["records_reencrypted"] == 0
    assert rt.unwrap_dek(r["wrapped"][0], b"L" * 32)["dek"] == d
    with pytest.raises(ValueError):
        rt.unwrap_dek(r["wrapped"][0], KEK)


def test_secrtt_dek_rotation_does_reencrypt():
    d1 = rt.generate_dek(b"seed", b"row-1")["dek"]
    d2 = rt.generate_dek(b"seed", b"row-2")["dek"]
    sealed = rt.seal_record(b"salary: 42", d1, b"record-nonce"[:12])
    out = rt.rotate_dek(sealed, d1, d2, b"another-nonc"[:12])
    assert out["records_reencrypted"] == 1
    assert rt.open_record(out["sealed"], d2) == b"salary: 42"


def test_secrtt_rotation_cost_ratio():
    c = rt.rotation_cost(1000000, 4096)
    assert c["ratio"] == 4096.0 / 32.0
    assert c["records_touched_envelope"] == 0
    with pytest.raises(ValueError):
        rt.rotation_cost(0, 4096)


def test_secrtt_crypto_shred_reports_scope():
    d = rt.generate_dek(b"seed", b"row-1")["dek"]
    w1 = rt.wrap_dek(d, KEK, b"nonce-0-0001"[:12], "kek-1")
    w2 = rt.wrap_dek(d, b"L" * 32, b"new-nonce-01"[:12], "kek-2")
    r = rt.crypto_shred("kek-1", [w1, w2])
    assert r["records_shredded"] == 1
    assert r["still_recoverable"] == [1]
    assert not r["complete"]


def test_secrtt_refuses_a_reused_or_missing_nonce():
    d = rt.generate_dek(b"seed", b"row-1")["dek"]
    w = rt.wrap_dek(d, KEK, b"nonce-0-0001"[:12])
    with pytest.raises(ValueError):
        rt.rotate_kek([w, w], KEK, b"L" * 32,
                      [b"only-one-non"[:12]])
