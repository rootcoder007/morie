"""gaaspc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gaaspc import gaaspc


def test_gaaspc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gaaspc()
