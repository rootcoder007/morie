"""aggrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.aggrn import aggrn


def test_aggrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        aggrn()
