# morie.fn -- function file (rootcoder007/morie)
"""Gromov-Wasserstein discrepancy."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["gwdist", "ot_gromov_wasserstein"]


def gwdist(Cx, Cy, a, b, n_iter=50, epsilon=0.05, n_sinkhorn=50):
    """Gromov-Wasserstein discrepancy.

    min_T sum |C^X_ij - C^Y_kl|^2 T_ik T_jl   (Memoli 2011).

    Gromov-Wasserstein discrepancy between two metric measure spaces
    given only their internal distance matrices -- no common ambient
    space is needed, which is the point of the construction.

    The objective is quartic and its exact minimisation is NP-hard, so
    the coupling is refined by ``n_iter`` fixed entropic projected
    gradient steps from the product coupling, each an inner Sinkhorn
    loop of ``n_sinkhorn`` fixed iterations.  Iteration counts are fixed
    rather than tolerance-driven, so the result is reproducible; the
    value at the product coupling is returned alongside so the
    improvement is visible.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Gromov-Wasserstein discrepancy", payload=_c.gwdist(Cx=Cx, Cy=Cy, a=a, b=b, n_iter=n_iter, epsilon=epsilon, n_sinkhorn=n_sinkhorn))


ot_gromov_wasserstein = gwdist


def cheatsheet():
    return "otgw: Gromov-Wasserstein discrepancy"
