# morie.fn -- function file (rootcoder007/morie)
r"""TimesFM -- duplicate ledger entry.

The wave-3 ledger carries this method twice, as ``timesf`` and as
``timesfm``, both citing Das et al. (2024) and both describing the
same decoder-only foundation model. One paper, one method.

This module re-exports :mod:`timesfm` rather than duplicating it, so
the two entries cannot drift apart. The patching contract, the
input/output patch asymmetry and the rollout arithmetic are documented
there.

References
----------
Das, A., Kong, W., Sen, R. & Zhou, Y. (2024) "A decoder-only
foundation model for time-series forecasting", *Proceedings of the
41st International Conference on Machine Learning*, PMLR 235,
arXiv:2310.10688.

See Also
--------
:mod:`morie.fn.timesfm` -- the implementation.
"""

from .timesfm import (causal_mask, horizon_plan, input_patches,
                      rollout, rollout_steps)

__all__ = ["input_patches", "causal_mask", "rollout_steps", "rollout",
           "horizon_plan"]


def cheatsheet():
    from .timesfm import cheatsheet as _c
    return ("timesf: the same ledger method as `timesfm` -- one "
            "paper, one implementation, re-exported so the two "
            "entries cannot drift. " + _c())


# compact alias per ledger/NAMING.md
timesffoundation = rollout
