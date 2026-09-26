"""svcle is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svcle import coalition_equil


def test_svcle_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        coalition_equil(data=None)
