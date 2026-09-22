"""Tests for forsnp.forensic_lr."""

from morie.fn import _array_core as np

from morie.fn.forsnp import forensic_lr


def test_forsnp_basic():
    """Test basic functionality with two loci at theta=0 (HW product rule)."""
    # Locus 1: homozygote A5/A5 with frequency 0.2
    # Locus 2: heterozygote A3/A4 with frequencies 0.3 and 0.1
    genotype = [("A5", "A5"), ("A3", "A4")]
    freqs = [
        {"A5": 0.2},
        {"A3": 0.3, "A4": 0.1},
    ]

    result = forensic_lr(genotype, freqs, theta=0.0)

    assert isinstance(result, dict)

    # Documented return keys
    for key in ("rmp", "lr", "locus_rmp", "n_loci", "theta"):
        assert key in result, f"missing documented key: {key}"

    # Independent computation from the documented formula at theta=0:
    # homozygote -> p^2 ; heterozygote -> 2*p_i*p_j
    p_hom = 0.2
    p_i, p_j = 0.3, 0.1
    locus1 = p_hom * p_hom           # 0.04
    locus2 = 2.0 * p_i * p_j         # 0.06
    expected_rmp = locus1 * locus2   # 0.0024

    assert result["n_loci"] == 2
    assert result["theta"] == 0.0
    assert result["locus_rmp"][0] == locus1
    assert result["locus_rmp"][1] == locus2
    assert result["rmp"] == expected_rmp
    assert result["lr"] == 1.0 / expected_rmp


def test_forsnp_edge():
    """Test subpopulation correction (theta=0.01) using NRC II 4.10 formula."""
    # Single locus, heterozygote A3/A4 with frequencies 0.3 and 0.1
    genotype = [("A3", "A4")]
    freqs = [{"A3": 0.3, "A4": 0.1}]
    theta = 0.01

    result = forensic_lr(genotype, freqs, theta=theta)

    assert isinstance(result, dict)
    for key in ("rmp", "lr", "locus_rmp", "n_loci", "theta"):
        assert key in result, f"missing documented key: {key}"

    # Heterozygote formula from docstring:
    #   2 [t + (1-t) p_i][t + (1-t) p_j] / [(1+t)(1+2t)]
    t = theta
    p_i, p_j = 0.3, 0.1
    expected_locus = (
        2.0 * (t + (1.0 - t) * p_i) * (t + (1.0 - t) * p_j)
        / ((1.0 + t) * (1.0 + 2.0 * t))
    )
    expected_rmp = expected_locus  # single locus

    assert result["n_loci"] == 1
    assert result["theta"] == theta
    assert result["locus_rmp"][0] == expected_locus
    assert result["rmp"] == expected_rmp
    assert result["lr"] == 1.0 / expected_rmp
