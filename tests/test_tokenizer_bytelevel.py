"""Byte-level BPE (GGUF "gpt2" tokenizers: Llama 3, Qwen): a hand-built vocabulary, no model file.

Before 1.4.0 these vocabularies went through the SentencePiece path: "Hello world" came back as
"<|begin_of_text|>!!!Hello!!!world" from Llama 3.2 (the space byte is "Ġ" there, not "▁").
"""

from morie.tokenizer import Tokenizer, _bytes_to_unicode


class _FakeGGUF:
    def __init__(self, tokens, merges, types=None, bos=0):
        self._metadata = {
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.pre": "llama-bpe",
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.merges": merges,
            "tokenizer.ggml.token_type": types or [1] * len(tokens),
            "tokenizer.ggml.bos_token_id": bos,
            "tokenizer.ggml.eos_token_id": bos,
        }


def _tok():
    byte_chars = [_bytes_to_unicode()[b] for b in range(256)]
    merged = ["he", "ll", "hell", "hello", "Ġw", "Ġwo", "Ġwor", "Ġworl", "Ġworld"]
    tokens = ["<|bos|>", *byte_chars, *merged]
    merges = ["h e", "l l", "he ll", "hell o", "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d"]
    types = [3] + [1] * (len(tokens) - 1)  # 3: a control token
    return Tokenizer(gguf_model=_FakeGGUF(tokens, merges, types)), tokens


def test_merges_apply_in_rank_order_and_the_space_byte_is_g_dot():
    tok, tokens = _tok()
    ids = tok.encode("hello world")
    assert [tokens[i] for i in ids] == ["<|bos|>", "hello", "Ġworld"]
    assert tok.decode(ids) == "hello world"  # the control token is not text


def test_bytes_round_trip_and_the_byte_map_is_gpt2s():
    tok, _ = _tok()
    for s in ["héllo wörld", "a\tb\n\nc", "x  y", "€100", "日本"]:
        assert tok.decode(tok.encode(s)) == s
    m = _bytes_to_unicode()
    assert len(set(m.values())) == 256 and m[ord(" ")] == "Ġ" and m[ord("\n")] == "Ċ" and m[ord("A")] == "A"


def test_add_bos_false_leaves_it_out():
    tok, tokens = _tok()
    assert [tokens[i] for i in tok.encode("hello", add_bos=False)] == ["hello"]
