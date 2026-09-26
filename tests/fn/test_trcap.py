"""trcap is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trcap import trcap


def test_trcap_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trcap()
