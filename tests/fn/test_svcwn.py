"""svcwn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcwn import condorcet_winner


def test_svcwn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        condorcet_winner(data=None)
