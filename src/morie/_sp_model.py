# morie -- native SentencePiece .model parser (rootcoder007/morie)
"""Parse a SentencePiece ``.model`` file without the sentencepiece
package.

The file is a serialized ``ModelProto`` (sentencepiece_model.proto in
the google/sentencepiece repository). Protobuf wire format per the
Protocol Buffers encoding spec: a stream of (tag varint, payload)
where tag = (field_number << 3) | wire_type; wire type 0 = varint,
2 = length-delimited, 5 = 32-bit.

Fields read here (all this loader needs):

* ModelProto.pieces      -- field 1, repeated SentencePiece
* SentencePiece.piece    -- field 1, string
* SentencePiece.score    -- field 2, float (32-bit)
* SentencePiece.type     -- field 3, enum (1=NORMAL, 2=UNKNOWN,
                            3=CONTROL, 4=USER_DEFINED, 6=BYTE)
* TrainerSpec (field 2) is skipped; bos/eos ids follow the
  sentencepiece defaults when control pieces are present.
"""

from __future__ import annotations

import struct


def _varint(buf, i):
    out = 0
    shift = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, i
        shift += 7


def _fields(buf):
    """Yield (field_number, wire_type, value) over a protobuf buffer."""
    i = 0
    n = len(buf)
    while i < n:
        tag, i = _varint(buf, i)
        fnum, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 2:
            ln, i = _varint(buf, i)
            v = buf[i:i + ln]
            i += ln
        elif wt == 5:
            v = buf[i:i + 4]
            i += 4
        elif wt == 1:
            v = buf[i:i + 8]
            i += 8
        else:
            raise ValueError("unsupported protobuf wire type %d" % wt)
        yield fnum, wt, v


def load_model(path):
    """Return (pieces, scores, types, bos_id, eos_id, unk_id)."""
    buf = open(path, "rb").read()
    pieces, scores, types = [], [], []
    for fnum, wt, v in _fields(buf):
        if fnum == 1 and wt == 2:                 # SentencePiece
            piece, score, ptype = "", 0.0, 1
            for f2, w2, v2 in _fields(v):
                if f2 == 1 and w2 == 2:
                    piece = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 5:
                    (score,) = struct.unpack("<f", v2)
                elif f2 == 3 and w2 == 0:
                    ptype = v2
            pieces.append(piece)
            scores.append(float(score))
            types.append(int(ptype))
    unk_id = next((i for i, t in enumerate(types) if t == 2), 0)
    # sentencepiece convention: control pieces <s> and </s>
    bos_id = next((i for i, p in enumerate(pieces) if p == "<s>"), 1)
    eos_id = next((i for i, p in enumerate(pieces) if p == "</s>"), 2)
    return pieces, scores, types, bos_id, eos_id, unk_id


def encode_unigram(text, pieces, scores, unk_id,
                   token_to_id=None):
    """Viterbi segmentation under the unigram LM (Kudo 2018,
    "Subword Regularization", ACL, eq. 3: the max-probability
    segmentation of the whitespace-escaped input).
    """
    if token_to_id is None:
        token_to_id = {p: i for i, p in enumerate(pieces)}
    s = "▁" + text.replace(" ", "▁")
    n = len(s)
    NEG = -1e18
    best = [NEG] * (n + 1)
    best[0] = 0.0
    back = [(-1, -1)] * (n + 1)
    maxlen = max((len(p) for p in pieces), default=1)
    for i in range(n):
        if best[i] <= NEG / 2:
            continue
        for j in range(i + 1, min(n, i + maxlen) + 1):
            tid = token_to_id.get(s[i:j])
            if tid is None:
                continue
            v = best[i] + scores[tid]
            if v > best[j]:
                best[j] = v
                back[j] = (i, tid)
    if best[n] <= NEG / 2:
        # force per-character with unk fallback
        ids = []
        for ch in s:
            ids.append(token_to_id.get(ch, unk_id))
        return ids
    ids = []
    j = n
    while j > 0:
        i, tid = back[j]
        if i < 0:                       # gap: emit unk for one char
            ids.append(unk_id)
            j -= 1
            continue
        ids.append(tid)
        j = i
    return ids[::-1]
