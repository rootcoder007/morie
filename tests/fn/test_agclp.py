"""agclp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.agclp import agclp


def test_agclp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        agclp()
