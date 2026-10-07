"""R's .rds / .RData read natively (no R, no rpy2): fixtures written by R 4.5 with saveRDS() / save().

d <- data.frame(i = c(1L, NA, 3L), x = c(1.5, NA, -2), s = c("a", NA, "\\u00e9"),
                f = factor(c("lo", "hi", NA)), b = c(TRUE, NA, FALSE), stringsAsFactors = FALSE)
"""

import base64
import math

import pytest

from morie import data
from morie.rds import read_rdata, read_rds

RDS = (
    "H4sIAAAAAAAAA12QQQrCMBBFf5NGaUERPIddu3PnCRTcxpqi0Fpoi7qzZ3HnLQQv5NKNRqbtVEgWyc+Q//5MNiEACX/gQSoAUOvVcjYH5JQu"
    "AEbNA8Cr0SzSYzoX77ZwfdE+vD3Qr0kLRUA+DSCw1lqgpoJ43gHJWPFH+43ubIPUnExadjDRVcX+wCrNHYeKU12yocckOq7yAsAXQOjM4oaq"
    "o84MIxQPwJHehUXJImGxbUnCOvnhTlc6SgqdGScrKPJzxHn0G4KastZ+qNkfOcUNOpgBAAA="
)
RDATA = (
    "H4sIAAAAAAAAA12QwWrCQBCGv+xmLQlUCn0OvXjx5qX0AYoFr1u7QSE2kEj1Vp+lN9+i0Bfy6EW3rMlY3DnM/Mwy3z+zL0+zUT7LAU3aS9AG"
    "wLxOnwdjSBWQkJKF6kA/hlfg/jIByY5LBN0PdXJsG1+HkO++f7jGQ+vSwiyQee897EJD/e5BC1b9o2926JXu05VNB1NdVy2WosoqmjDz0jYy"
    "cMUUdr6uauAM5NEtsan5sCsnCCMHiGWyFdGIKES8tSTtI//83a7tsKjtykVeWV1thuIXfkOFpbz3p27Z8x9V+QeDsgEAAA=="
)


def _check(df):
    assert list(df.columns) == ["i", "x", "s", "f", "b"]
    i, x = df["i"].tolist(), df["x"].tolist()
    assert i[0] == 1.0 and math.isnan(i[1]) and i[2] == 3.0
    assert x[0] == 1.5 and math.isnan(x[1]) and x[2] == -2.0
    assert df["s"].tolist() == ["a", None, "é"]
    assert df["f"].tolist() == ["lo", "hi", None]  # a factor reads as its labels
    assert df["b"].tolist() == [True, None, False]


def test_rds_data_frame_matches_what_r_wrote(tmp_path):
    p = tmp_path / "d.rds"
    p.write_bytes(base64.b64decode(RDS))
    _check(read_rds(p))


def test_rdata_workspace_by_name(tmp_path):
    p = tmp_path / "d.RData"
    p.write_bytes(base64.b64decode(RDATA))
    objs = read_rdata(p)
    assert list(objs) == ["e"]
    _check(objs["e"])


def test_load_dataset_reads_an_r_object_instead_of_asking_for_r(tmp_path):
    p = tmp_path / "d.rds"
    p.write_bytes(base64.b64decode(RDS))
    _check(data._r_object_frame(p))
    bad = tmp_path / "x.rds"
    bad.write_bytes(b"not R")
    assert data._r_object_frame(bad) is None  # not R data: the saved-for-R path still applies


def test_not_r_data_is_refused_with_a_reason(tmp_path):
    p = tmp_path / "x.rds"
    p.write_bytes(b"A\n")
    with pytest.raises(ValueError, match="not XDR-serialized R data"):
        read_rds(p)
