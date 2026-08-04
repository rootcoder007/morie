"""Tests for the native Parquet codec.

Real store files are used where they are present on the machine; the
round-trip tests are self-contained so the suite still means something
on a bare checkout.
"""
import datetime
import glob
import os

import pytest

from morie.fn._frame_core import DataFrame, read_parquet
from morie.fn import _parquet_core as P

STORE = sorted(glob.glob(os.path.expanduser(
    "~/.cache/R/rmoriedata/*.parquet")))


def _frame():
    return DataFrame({
        "case": ["23-OCI-001", "23-OCI-002", None, "23-OFP-9"],
        "year": [2023, 2023, 2024, 2024],
        "rate": [0.5, -1.25, None, 3.75],
        "closed": [True, False, True, None],
    })


@pytest.mark.parametrize("compression", ["snappy", None])
def test_round_trip_preserves_every_cell(tmp_path, compression):
    df = _frame()
    p = str(tmp_path / "t.parquet")
    P.to_parquet(df, p, compression=compression)
    back = P.read_parquet(p)
    assert list(back.columns) == list(df.columns)
    for c in df.columns:
        assert list(back[c]) == list(df[c]), c


def test_file_is_framed_as_parquet(tmp_path):
    p = str(tmp_path / "t.parquet")
    P.to_parquet(_frame(), p)
    raw = open(p, "rb").read()
    assert raw[:4] == b"PAR1" and raw[-4:] == b"PAR1"


def test_column_subset_skips_the_rest(tmp_path):
    p = str(tmp_path / "t.parquet")
    P.to_parquet(_frame(), p)
    sub = P.read_parquet(p, columns=["year"])
    assert list(sub.columns) == ["year"]
    assert list(sub["year"]) == [2023, 2023, 2024, 2024]


def test_unknown_column_is_refused(tmp_path):
    p = str(tmp_path / "t.parquet")
    P.to_parquet(_frame(), p)
    with pytest.raises(KeyError, match="nope"):
        P.read_parquet(p, columns=["nope"])


def test_not_a_parquet_file_is_refused(tmp_path):
    p = tmp_path / "x.parquet"
    p.write_bytes(b"definitely not parquet")
    with pytest.raises(ValueError, match="PAR1"):
        P.read_parquet(str(p))


def test_logical_types_survive_the_writer(tmp_path):
    # A timestamp column that came back as a bare number after a
    # round-trip was a real defect: it stayed readable but stopped
    # being a timestamp to any other engine.
    df = DataFrame({
        "d": [datetime.date(2023, 5, 1), datetime.date(2024, 1, 31)],
        "t": [datetime.datetime(2021, 5, 21, 5, 1,
                                tzinfo=datetime.timezone.utc),
              datetime.datetime(2022, 6, 2, 12, 0,
                                tzinfo=datetime.timezone.utc)],
    })
    p = str(tmp_path / "t.parquet")
    P.to_parquet(df, p)
    back = P.read_parquet(p)
    assert list(back["d"]) == list(df["d"])
    assert list(back["t"]) == list(df["t"])


def test_snappy_round_trips_including_overlapping_copies():
    # Long runs are exactly what snappy encodes with an overlapping
    # copy, which is the one branch a naive decompressor gets wrong.
    for payload in (b"", b"a", b"a" * 100000,
                    (b"abcdefgh" * 5000) + b"tail"):
        assert P._snappy_decompress(P._snappy_compress(payload)) == payload


def test_frame_core_entry_points_are_wired(tmp_path):
    p = str(tmp_path / "t.parquet")
    _frame().to_parquet(p)
    assert len(read_parquet(p)) == 4


@pytest.mark.skipif(not STORE, reason="rmoriedata store not on this box")
@pytest.mark.parametrize("path", STORE)
def test_real_store_files_decode(path):
    df = P.read_parquet(path)
    assert len(df) > 0 and len(df.columns) > 0
    # every column is the full height -- a short column would mean the
    # definition levels were mis-decoded
    for c in df.columns:
        assert len(list(df[c])) == len(df)
