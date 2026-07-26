"""Tests for gmatv.grm_vanraden.

Spec: Montesinos-Lopez et al., Multivariate Statistical Machine Learning
Methods for Genomic Prediction, Sec. 2.4 pp. 49-52, after VanRaden (2008),
J Dairy Sci 91(11):4414-4423.

Every expected value here is hand-transcribed from the typeset PDF, never
produced by running grm_vanraden. The text extraction of this section is
unusable for the purpose -- it renders the formulae as `G ¼ 1 p XX T` and
`zij ¼ xij  2p j = 2p j 1  p j`, losing the fractions, signs and radicals --
so the PDF is the source of truth and the txt is only a search index.
"""

import json
import pathlib

import numpy as np
import pytest

from morie.fn.gmatv import grm_vanraden

_FIXTURE = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "gmatv.json").read_text()
)
X = np.array(_FIXTURE["input"]["X_markers_8x7"], dtype=float)


def test_transcription_matches_printed_allele_frequencies():
    """Validate the transcribed matrix independently of any G.

    The book prints phat alongside the worked example; reproducing it proves
    the matrix was copied correctly without relying on a method being right.
    """
    assert np.allclose(
        X.mean(axis=0) / 2.0, _FIXTURE["transcription_check"]["printed_phat"]
    )


def test_gvr1_matches_book_worked_example():
    """G_VR1 vs the printed table.

    VanRaden Method 1 (ZZ'/2sum-pq, primary p.4416) == Montesinos Method 2
    (secondary p.51). The two sources number the methods differently; the
    aliases follow the primary.
    """
    got = np.asarray(grm_vanraden(X, method="G_VR1")["estimate"], dtype=float)
    assert np.allclose(np.round(got, 3), _FIXTURE["method2"]["printed_G"], atol=5e-4)


def test_gxx_uses_the_marker_divisor_not_the_line_divisor():
    """G_XX = XX'/m, per the printed formula AND the printed R code.

    The source's printed numeric table for this method is reproduced only by
    dividing by the number of lines; both its formula and its `dim(X)[2]` R
    code divide by markers. We follow the formula. This test pins that
    decision so a future "fix" toward the printed table fails loudly.
    """
    got = np.asarray(grm_vanraden(X, method="G_XX")["estimate"], dtype=float)
    n, m = X.shape
    assert np.allclose(got, X @ X.T / m)
    # And is demonstrably NOT the printed (line-divisor) table.
    assert not np.allclose(np.round(got, 3), _FIXTURE["method1"]["printed_G_row1"][0])


def test_gvr2_uses_allele_frequency_scaling_not_sample_sd():
    """G_VR2 (VanRaden Method 2, ZDZ') scales by sqrt(2p(1-p)), per the printed formula.

    The source's printed table for this method is reproduced by its R code's
    sample-SD scaling instead, differing from the formula by up to 0.162.
    Allele-frequency scaling is the defining feature of VanRaden Method 3;
    sample-SD scaling is a different quantity. Convention drift recorded in
    the worklist.
    """
    got = np.asarray(grm_vanraden(X, method="G_VR2")["estimate"], dtype=float)
    p = X.mean(axis=0) / 2.0
    Z = (X - 2.0 * p) / np.sqrt(2.0 * p * (1.0 - p))
    assert np.allclose(got, Z @ Z.T / X.shape[1])

    sd = (X - X.mean(0)) / X.std(0, ddof=1)
    alt = sd @ sd.T / X.shape[1]
    assert np.max(np.abs(got - alt)) > 0.1, "the two readings must remain distinguishable"


@pytest.mark.parametrize("method", [1, 2, 3, "G_VR1", "G_VR2", "G_XX"])
def test_identity_symmetric_and_psd(method):
    """Second identity test: a GRM is a Gram matrix, so symmetric and PSD.

    G = Z Z' / c with c > 0 is positive semi-definite by construction, for
    every method. A sign error, a bad divisor, or a transpose slip breaks one
    of these two properties while still returning plausible-looking numbers.
    """
    G = np.asarray(grm_vanraden(X, method=method)["estimate"], dtype=float)
    assert np.allclose(G, G.T, atol=1e-12)
    eig = np.linalg.eigvalsh(G)
    assert eig.min() > -1e-8, f"not PSD: min eigenvalue {eig.min():.3e}"


def test_identity_trace_over_n_tracks_average_relationship():
    """trace(G)/n is the mean genomic self-relationship.

    For G_VR1 and G_VR2, which are centred on the allele frequencies, this
    sits near 1 for a population in Hardy-Weinberg equilibrium. G_XX is
    uncentred so it carries no such normalisation -- asserting the same
    bound on G_XX would be wrong, and that asymmetry is the point.
    """
    for method in ("G_VR1", "G_VR2"):
        G = np.asarray(grm_vanraden(X, method=method)["estimate"], dtype=float)
        assert 0.3 < np.trace(G) / X.shape[0] < 2.5


def test_rejects_1d_input():
    """The test used to pass a 1-D array; the function was right to refuse.

    Fixing the test rather than loosening the function: a genomic relationship
    matrix is not defined for a single vector of genotypes.
    """
    with pytest.raises(ValueError, match="2D"):
        grm_vanraden(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="method must be one of"):
        grm_vanraden(X, method=99)
