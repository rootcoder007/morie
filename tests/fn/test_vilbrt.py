"""Tests for vilbrt (alias of hmvilb.geron_vilbert)."""

from morie.fn.hmvilb import geron_vilbert
from morie.fn.vilbrt import vilbert_two_stream, vilbrt

IMG = [[0.5, 0.1, 0.2], [0.3, 0.7, 0.1]]
TXT = [[0.2, 0.4, 0.6], [0.9, 0.1, 0.3], [0.1, 0.1, 0.8]]


def _rows(m):
    return [[float(v) for v in row] for row in m]


def test_vilbrt_anchor_coattention():
    # Lu et al. (2019) Sec 3.1: co-attention swaps queries across
    # modalities, so attention_v2t is (regions x tokens) and its rows
    # are a softmax (sum to 1); attention_t2v is (tokens x regions).
    r = vilbrt(IMG, TXT, d_model=4, seed=0)
    a_v2t = _rows(r["attention_v2t"])
    a_t2v = _rows(r["attention_t2v"])
    assert len(a_v2t) == 2 and len(a_v2t[0]) == 3
    assert len(a_t2v) == 3 and len(a_t2v[0]) == 2
    for row in a_v2t + a_t2v:
        assert abs(sum(row) - 1.0) < 1e-12


def test_vilbrt_alias_exact_zero():
    a = vilbrt(IMG, TXT, d_model=4, seed=3)
    b = geron_vilbert(IMG, TXT, d_model=4, seed=3)
    assert _rows(a["attention_v2t"]) == _rows(b["attention_v2t"])
    assert _rows(a["attention_t2v"]) == _rows(b["attention_t2v"])
    assert a["estimate"] == b["estimate"]
    assert vilbert_two_stream is vilbrt
