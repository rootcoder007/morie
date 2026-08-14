# morie.fn -- function file (rootcoder007/morie)
r"""BLIP-2 Q-Former -- re-export of :mod:`blip2v`.

``blipqf`` and ``blip2v`` are two ledger rows citing the same paper
(Li, Li, Savarese & Hoi 2023). They are kept as one implementation
with a re-export so the two entries cannot drift apart, exactly as
``timesf`` re-exports ``timesfm`` and ``egcn`` re-exports ``egnnL``.

See :mod:`blip2v` for the mechanism, the two-stage argument and the
references.
"""

from .blip2v import (cheatsheet, project_to_llm, qformer_attend,
                     query_tokens, stage_one_objectives,
                     trainable_fraction)

__all__ = ["query_tokens", "qformer_attend", "trainable_fraction",
           "stage_one_objectives", "project_to_llm", "cheatsheet"]

# compact alias per ledger/NAMING.md
queryingtransformer = qformer_attend
