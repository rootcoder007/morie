"""trcrp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.trcrp import trcrp


def test_trcrp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        trcrp()
