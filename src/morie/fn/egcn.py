# morie.fn -- function file (rootcoder007/morie)
r"""E(n)-equivariant graph convolution -- re-export of :mod:`egnnL`.

``egcn`` and ``egnnL`` are two ledger rows citing the same paper
(Satorras, Hoogeboom & Welling 2021). They are kept as one
implementation with a re-export so the two entries cannot drift apart,
exactly as ``timesf`` re-exports ``timesfm``.

See :mod:`egnnL` for the equations, the equivariance argument and the
references.
"""

from .egnnL import (cheatsheet, coord_update, edge_message, egcl,
                    equivariance_error, run_egnn)

__all__ = ["edge_message", "coord_update", "egcl", "run_egnn",
           "equivariance_error", "cheatsheet"]

# compact alias per ledger/NAMING.md
equivariantgraphconv = run_egnn

# public names resolved by fn/_lazy_map.json
e_gcn = run_egnn
