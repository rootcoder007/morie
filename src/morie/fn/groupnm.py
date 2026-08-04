# morie.fn -- function file (rootcoder007/morie)
"""Group normalisation."""

from ._richresult import RichResult
from . import _unclrcore as _c

__all__ = ["grpnorm", "groupnorm", "group_norm"]


def grpnorm(x, n_groups, eps=1e-05):
    """Group normalisation.

    Group normalisation over channel groups.

    Wu & He (2018).  Channels are split into ``n_groups`` groups and each
    group is standardised over its own elements.  Unlike batch norm the
    statistics come from one sample, so the result does not depend on
    batch size -- which is why it holds up at batch size 1.
    ``x`` is a flat channel-major vector of length C * S.

    Returns
    -------
    RichResult
        Inherits from ``dict``; keys are listed above.
    """
    return RichResult(title="Group normalisation", payload=_c.grpnorm(x=x, n_groups=n_groups, eps=eps))


group_norm = grpnorm


def cheatsheet():
    return "groupnm: Group normalisation"


# compact alias per ledger/NAMING.md (pre-existing spelling, kept working)
groupnorm = grpnorm
