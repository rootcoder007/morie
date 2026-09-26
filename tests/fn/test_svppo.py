"""svppo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svppo import party_position


def test_svppo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        party_position(data=None)
