"""Tokenizer for MORIE's inference engine.

Loads tokenization data from GGUF model metadata (the ``tokenizer.ggml.*``
keys) and provides encode/decode without requiring SentencePiece at runtime.

Falls back to SentencePiece if a ``.model`` file is provided explicitly.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path


@functools.lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-character map (Radford et al. 2019, encoder.py)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


# the pre-tokeniser of llama.cpp's "llama-bpe" (Llama 3) / tiktoken cl100k, with \p{L} written as
# [^\W\d_] and \p{N} as \d (the stdlib re has no Unicode property classes)
_PRE_LLAMA3 = re.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?[^\W\d_]+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# GPT-2's own pattern, for "gpt2"-model vocabularies with another pre-tokeniser
_PRE_GPT2 = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")


class Tokenizer:
    """BPE / SentencePiece tokenizer loaded from GGUF metadata.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to a GGUF file (reads ``tokenizer.ggml.*`` metadata) or a
        SentencePiece ``.model`` file.
    gguf_model : GGUFModel, optional
        An already-loaded :class:`morie.gguf_loader.GGUFModel` instance.

    Examples
    --------
    Needs a GGUF model file on disk (the path below is a placeholder):

    >>> from morie.gguf_loader import GGUFModel
    >>> model = GGUFModel("path/to/model.gguf")  # doctest: +SKIP
    >>> tok = Tokenizer(gguf_model=model)  # doctest: +SKIP
    >>> ids = tok.encode("Hello world")  # doctest: +SKIP
    >>> tok.decode(ids)  # doctest: +SKIP
    'Hello world'
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        gguf_model=None,
    ):
        self._vocab: list[str] = []
        self._scores: list[float] = []
        self._token_to_id: dict[str, int] = {}
        self._merges: list[tuple[str, str]] = []
        self._eos_id: int = 2
        self._bos_id: int = 1
        self._sp = None  # SentencePiece processor (optional fallback)
        self._byte_level = False  # GPT-2 byte-level BPE (Llama 3, Qwen, ...)
        self._control: set[int] = set()

        if gguf_model is not None:
            self._load_from_gguf(gguf_model)
        elif model_path is not None:
            path = Path(model_path)
            if path.suffix == ".model":
                self._load_sentencepiece(path)
            else:
                from .gguf_loader import GGUFModel

                gm = GGUFModel(path)
                self._load_from_gguf(gm)
        else:
            raise ValueError("Provide either model_path or gguf_model")

    def _load_from_gguf(self, gm) -> None:
        """Extract tokenizer from GGUF metadata keys."""
        meta = gm._metadata

        # Vocabulary tokens
        tokens = meta.get("tokenizer.ggml.tokens", [])
        if not tokens:
            raise ValueError("GGUF file has no tokenizer.ggml.tokens metadata")

        self._vocab = [t if isinstance(t, str) else t.decode("utf-8", errors="replace") for t in tokens]
        self._scores = meta.get("tokenizer.ggml.scores", [0.0] * len(self._vocab))
        self._token_to_id = {tok: i for i, tok in enumerate(self._vocab)}

        # Special tokens
        self._bos_id = meta.get("tokenizer.ggml.bos_token_id", 1)
        self._eos_id = meta.get("tokenizer.ggml.eos_token_id", 2)

        # BPE merges (if present)
        merges_raw = meta.get("tokenizer.ggml.merges", [])
        self._merges = []
        for m in merges_raw:
            m = m if isinstance(m, str) else m.decode("utf-8", errors="replace")
            parts = m.split(" ", 1)
            if len(parts) == 2:
                self._merges.append((parts[0], parts[1]))
        # "gpt2": byte-level BPE with ranked merges (Llama 3 and most current models); "llama": SentencePiece
        self._byte_level = meta.get("tokenizer.ggml.model") == "gpt2" and bool(self._merges)
        self._ranks = {pair: r for r, pair in enumerate(self._merges)}
        self._pre = (
            _PRE_LLAMA3
            if meta.get("tokenizer.ggml.pre", "llama-bpe") in ("llama-bpe", "llama3", "default")
            else _PRE_GPT2
        )
        # control tokens (<|begin_of_text|>, <|eot_id|>, ...) are not text: decode leaves them out
        types = meta.get("tokenizer.ggml.token_type", [])
        self._control = {i for i, t in enumerate(types) if t == 3}
        self._add_bos = bool(meta.get("tokenizer.ggml.add_bos_token", True))

    def _load_sentencepiece(self, path: Path) -> None:
        """Load from a SentencePiece .model file."""
        from morie._sp_model import load_model

        pieces, scores, types, bos, eos, unk = load_model(str(path))
        del types
        self._sp = None  # native path only
        self._vocab = pieces
        self._scores = scores
        self._token_to_id = {tok: i for i, tok in enumerate(self._vocab)}
        self._bos_id = bos
        self._eos_id = eos
        self._unk_id = unk
        self._unigram = True  # Viterbi encode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str, *, add_bos: bool = True) -> list[int]:
        """Encode text to token IDs.

        Uses SentencePiece if available, otherwise greedy BPE matching.
        """
        if getattr(self, "_unigram", False):
            from morie._sp_model import encode_unigram

            ids = encode_unigram(text, self._vocab, self._scores, getattr(self, "_unk_id", 0), self._token_to_id)
            if add_bos and (not ids or ids[0] != self._bos_id):
                ids = [self._bos_id] + ids
            return ids

        if self._byte_level:
            ids = []
            enc = _bytes_to_unicode()
            for chunk in self._pre.findall(text):
                word = "".join(enc[b] for b in chunk.encode("utf-8"))
                ids.extend(self._token_to_id[t] for t in self._bpe_merge(word))
            return [self._bos_id, *ids] if add_bos and self._add_bos else ids

        # Greedy byte-fallback encoding for GGUF vocab
        tokens = self._bpe_encode(text)
        ids = [self._token_to_id.get(t, 0) for t in tokens]
        if add_bos:
            ids = [self._bos_id] + ids
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        if self._sp is not None:
            return self._sp.Decode(ids)

        if self._byte_level:
            dec = {c: b for b, c in _bytes_to_unicode().items()}
            raw = bytearray()
            for i in ids:
                if 0 <= i < len(self._vocab) and i not in self._control:
                    raw.extend(dec[c] for c in self._vocab[i] if c in dec)
            return raw.decode("utf-8", errors="replace")

        pieces = []
        for i in ids:
            if 0 <= i < len(self._vocab):
                piece = self._vocab[i]
                # SentencePiece-style: ▁ means space
                piece = piece.replace("▁", " ")
                # Byte tokens: <0xHH>
                if piece.startswith("<0x") and piece.endswith(">"):
                    try:
                        pieces.append(chr(int(piece[3:-1], 16)))
                        continue
                    except ValueError:
                        pass
                pieces.append(piece)
        text = "".join(pieces)
        # Strip leading space if BOS was decoded
        if text.startswith(" "):
            text = text[1:]
        return text

    def _bpe_merge(self, word: str) -> list[str]:
        """Byte-level BPE on one pre-token: repeatedly merge the adjacent pair of lowest rank."""
        if word in self._token_to_id:
            return [word]
        parts = list(word)
        while len(parts) > 1:
            best, at = None, -1
            for k in range(len(parts) - 1):
                r = self._ranks.get((parts[k], parts[k + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, k
            if best is None:
                break
            parts[at : at + 2] = [parts[at] + parts[at + 1]]
        return parts

    def _bpe_encode(self, text: str) -> list[str]:
        """Greedy longest-match encoding with UTF-8 byte fallback."""
        # SentencePiece convention: leading space becomes ▁
        text = "▁" + text.replace(" ", "▁")
        tokens = []
        i = 0
        while i < len(text):
            # Greedy: find longest vocab match starting at position i
            best = None
            best_len = 0
            for length in range(min(32, len(text) - i), 0, -1):
                candidate = text[i : i + length]
                if candidate in self._token_to_id:
                    best = candidate
                    best_len = length
                    break
            if best is not None:
                tokens.append(best)
                i += best_len
            else:
                # Byte fallback: encode character as UTF-8 byte tokens
                char_bytes = text[i].encode("utf-8")
                for b in char_bytes:
                    byte_tok = f"<0x{b:02X}>"
                    tokens.append(byte_tok)
                i += 1
        return tokens

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        if self._sp is not None:
            return self._sp.GetPieceSize()
        return len(self._vocab)

    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return self._eos_id

    @property
    def bos_id(self) -> int:
        """Beginning-of-sequence token ID."""
        return self._bos_id

    def __repr__(self) -> str:
        src = "sentencepiece" if self._sp else "gguf"
        return f"Tokenizer(vocab_size={self.vocab_size}, source={src})"
