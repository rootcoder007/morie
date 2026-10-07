# morie.fn -- function file from book-equation translation pipeline (rootcoder007/morie)
"""Aldrich-McKelvey scaling for spatial voting."""

from __future__ import annotations

from ._containers import DescriptiveResult


def aldrich_mckelvey_scaling(
    Z,
    n_dims: int = 1,
) -> DescriptiveResult:
    """Aldrich-McKelvey scaling of perceptual data (closed form).

    :param Z: Respondent x stimulus placement matrix.
    :param n_dims: Number of latent dimensions (must be 1).
    :return: DescriptiveResult; ``value`` is the stimulus positions, the
        full result is in ``extra``.

    .. epigraph:: Give me a place to stand and I will move the earth. -- Archimedes
    """
    from morie._spatial_voting import aldrich_mckelvey as _fn

    result = _fn(Z, n_dims=n_dims)
    return DescriptiveResult(
        name="aldrich_mckelvey_scaling",
        value=[float(v) for v in result["zhat"]],
        extra=result,
    )


amscl = aldrich_mckelvey_scaling


def cheatsheet() -> str:
    return "aldrich_mckelvey_scaling({}) -> Aldrich-McKelvey scaling for spatial voting."
