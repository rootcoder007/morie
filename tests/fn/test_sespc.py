"""sespc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sespc import sespc


def test_sespc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sespc()
