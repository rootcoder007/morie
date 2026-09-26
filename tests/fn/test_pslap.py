"""pslap is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.pslap import pslap


def test_pslap_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        pslap()
