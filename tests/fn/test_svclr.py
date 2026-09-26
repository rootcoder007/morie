"""svclr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svclr import condorcet_loser


def test_svclr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        condorcet_loser(data=None)
