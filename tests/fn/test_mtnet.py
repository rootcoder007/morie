"""mtnet is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtnet import mtnet


def test_mtnet_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtnet()
