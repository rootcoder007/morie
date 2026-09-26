"""simcs is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.simcs import simcs


def test_simcs_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        simcs()
