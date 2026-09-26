"""svppc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svppc import party_congress


def test_svppc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        party_congress(data=None)
