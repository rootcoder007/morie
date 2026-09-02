"""mqtmpl positions are centiMorgans; every Haldane call converts to Morgans.
The anchor is shared with the two R arms (tests/testthat/test-never-run-fixes.R)."""
import math
from morie.fn import mqtmpl


def test_em_scan_anchor_cm_positions():
    y = [1.9, 0.3, 2.4, 0.1, 2.2, 0.6, 1.7, 0.4, 2.0, 0.2, 2.6, 0.5]
    mk = [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
          [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
          [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]]
    r = mqtmpl.scanone(y, mk, [0.0, 10.0, 20.0], method="em", step=5)
    assert r["peak_position"] == 0.0
    assert math.isclose(r["peak_lod"], 6.89523009508, rel_tol=1e-9)
    want = [6.89523009508, 6.28234432026, 1.84002019604, 1.84002019604,
            1.1367051393, 0.295401178129]
    assert all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(r["lod"], want))


def test_cm_not_morgans():
    # 10 cM between adjacent markers is r ~ 0.09, not r ~ 0.5 (10 Morgans)
    y = [1.9, 0.3, 2.4, 0.1, 2.2, 0.6, 1.7, 0.4, 2.0, 0.2, 2.6, 0.5]
    mk = [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]]
    r = mqtmpl.scanone(y, mk, [0.0, 10.0], method="em", step=5)
    # the mid-interval LOD stays close to the flanking values under cM;
    # under Morgans the interval would be unlinked and collapse toward 0
    assert r["lod"][1] > 0.5 * min(r["lod"][0], r["lod"][2])
