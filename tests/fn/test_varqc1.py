"""Tests for varqc1.vcf_filter (GATK hard filters)."""

import pytest

from morie.fn.varqc1 import vcf_filter


F = ["QD", "QUAL", "SOR", "FS", "MQ", "MQRankSum", "ReadPosRankSum"]
GATK_SNP = [("QD", "lt", 2.0), ("QUAL", "lt", 30.0), ("SOR", "gt", 3.0), ("FS", "gt", 60.0),
            ("MQ", "lt", 40.0), ("MQRankSum", "lt", -12.5), ("ReadPosRankSum", "lt", -8.0)]


def _label(row, thr, fields):
    """A record fails every threshold it crosses; the FILTER string lists
    them in threshold order joined by ';', PASS when none."""
    bad = []
    for name, op, cut in thr:
        v = row[fields.index(name)]
        if (op == "lt" and v < cut) or (op == "gt" and v > cut):
            bad.append("%s%s%s" % (name, "<" if op == "lt" else ">", ("%g" % cut)))
    return ";".join(bad) if bad else "PASS"


def test_varqc1_basic():
    """Default SNP thresholds are GATK's hard-filter recommendations, and
    each record's FILTER string is its list of crossed thresholds."""
    V = [[10, 50, 1, 5, 60, 0, 0], [1.5, 50, 1, 5, 60, 0, 0], [10, 50, 4, 70, 60, 0, 0],
         [10, 20, 1, 5, 30, -13, -9], [2.0, 30, 3, 60, 40, -12.5, -8]]
    r = vcf_filter(V, fields=F)
    assert [tuple(t) for t in r["thresholds"]] == GATK_SNP
    assert r["filter"] == [_label(row, GATK_SNP, F) for row in V]
    # thresholds are strict: values exactly at the cut-offs pass
    assert r["filter"][4] == "PASS"
    assert r["n_pass_hard"] == 2
    assert r["estimate"] == pytest.approx(2 / 5, abs=1e-15)


def test_varqc1_edge():
    """Indel mode uses the looser indel thresholds; fields are required."""
    r = vcf_filter([[10, 50, 5, 100, 60, 0, 0]], fields=F, mode="indel")
    assert r["filter"] == ["PASS"]
    with pytest.raises(ValueError):
        vcf_filter([[1, 2]])
