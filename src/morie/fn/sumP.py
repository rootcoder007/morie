# morie.fn -- function file (rootcoder007/morie)
"""Graph readout by sum pooling."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["sumpl", "sumpool", "sum_pool"]


def sumpl(H):
    """Graph readout by sum pooling.

    Graph readout by summing node embeddings: h_G = sum_v h_v.

    Sum pooling is the readout that makes a message-passing network as
    discriminative as the Weisfeiler-Lehman test (Xu et al. 2019, GIN);
    mean and max pooling both collapse multisets that sum pooling keeps
    apart, so all three are returned for comparison.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Graph readout by sum pooling", payload=_c.sumpl(H=H))


sum_pool = sumpl


def cheatsheet():
    return "sumP: Graph readout by sum pooling"


# compact alias per ledger/NAMING.md (pre-existing spelling, kept working)
sumpool = sumpl
