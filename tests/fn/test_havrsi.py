"""havrsi is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.havrsi import havrsi


def test_havrsi_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        havrsi()
