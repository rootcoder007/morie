"""clgap is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clgap import clgap


def test_clgap_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clgap()
