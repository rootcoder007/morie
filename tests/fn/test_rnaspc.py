"""rnaspc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.rnaspc import rnaspc


def test_rnaspc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        rnaspc()
